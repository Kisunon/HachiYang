"""阶段 1/2 三阶段流水线测试（Phase 3.25，零 API 配额）。

对齐 RAG_SYSTEM_DESIGN §3.1.1 与验收项 R-17：
- 阶段 1：EPUB 解析 + 多卷合订本按卷拆分 → extracted.md / extracted.json 落盘；
- 阶段 2：extracted.json → chunks.json，6 级溯源元数据 + 原文锚点；
- 卷检测三种模式：弗洛伊德式（h1 卷名 + 编委会重复）、哲学通史式（版权页重复）、
  荣格式（无卷信号 → 整书单卷降级）。

全部使用合成 EPUB（ebooklib 临时生成），不依赖 knowledge_base 真实文件、不调用任何 API。
真实书端到端验证由 scripts/run_pipeline_stages.py 人工执行。
"""

from __future__ import annotations

import json
from pathlib import Path

import ebooklib
import pytest
from bs4 import BeautifulSoup
from ebooklib import epub

from app.rag.chunker import chunk_extracted_volume, run_chunk_stage
from app.rag.extractor import (
    _build_volume_tree,
    _detect_boundaries,
    _Elem,
    extract_book,
    run_extract_stage,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==================== 合成 EPUB 工具 ====================

def _make_epub(path: Path, title: str, html_parts: list[str]) -> Path:
    """把若干段 HTML 拼为一个 EPUB（每 part 一个 chapter 文档，保持顺序）。"""
    book = epub.EpubBook()
    book.set_identifier(f"id-{title}")
    book.set_title(title)
    book.set_language("zh")
    chapters = []
    for i, html in enumerate(html_parts):
        c = epub.EpubHtml(
            title=f"doc{i}", file_name=f"doc_{i}.xhtml", lang="zh",
        )
        c.content = f"<html><body>{html}</body></html>"
        book.add_item(c)
        chapters.append(c)
    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + chapters
    epub.write_epub(str(path), book)
    return path


def _p(text: str) -> str:
    return f"<p>{text}</p>"


def _h(level: int, text: str) -> str:
    return f"<h{level}>{text}</h{level}>"


# 足量正文（>100 字），避免空区间过滤
_BODY1 = "精神分析学认为，人类的心理活动可以分为意识与无意识两个层次。" * 6
_BODY2 = "力比多是一种驱动个体寻求快感的心理能量，它在不同发展阶段有不同表现。" * 6
_BODY3 = "梦的本质是被压抑欲望的象征性满足，显梦与隐梦之间存在稽查作用。" * 6
_BODY4 = "移情是指来访者将早期关系模式转移到治疗师身上的临床现象。" * 6


# ==================== 阶段 1：卷检测单元测试 ====================

def _stream(items: list[tuple]) -> list[_Elem]:
    """把 (kind, level, text) 元组列表转为 _Elem 流。"""
    elems = []
    for kind, *rest in items:
        if kind == "h":
            level, text = rest
            elems.append(_Elem(kind="h", level=level, text=text))
        else:
            text = rest[0]
            elems.append(_Elem(kind="p", level=0, text=text))
    return elems


def test_detect_freud_style_volumes():
    """弗洛伊德式：h1 卷名 + h2 编委会/编译说明重复 → C 类边界，多卷拆分。"""
    elems = _stream([
        ("h", 1, "版权信息"),
        ("p", "本书版权归出版社所有。出版发行：某某出版社。版次印次字数定价ISBN。"),
        ("h", 1, "爱情心理学"),
        ("h", 2, "《弗洛伊德文集》编委会"),
        ("p", "主编某某，副主编某某，编委某某某等人员名单如下所列。"),
        ("h", 2, "新版编译说明"),
        ("h", 3, "一、版本出处"),
        ("p", "本卷据某某版本译出，参考了多种权威译本。"),
        ("h", 2, "性学三论"),
        ("h", 3, "第一篇 性变态"),
        ("p", _BODY1),
        ("h", 2, "儿童性理论"),
        ("p", _BODY2),
        ("h", 1, "达·芬奇的童年回忆"),
        ("h", 2, "新版编译说明"),
        ("p", "本卷编译说明同上一版，版本出处与编译分工基本一致。"),
        ("h", 2, "戏剧中的变态人物"),
        ("p", _BODY3),
        ("h", 2, "作家与白日梦"),
        ("p", _BODY4),
        # 续编卷：h1 卷名但无编委会页（如《释梦（下）》直接接上卷章节）
        ("h", 1, "释梦（下）"),
        ("h", 2, "第六章 梦的工作"),
        ("p", _BODY1),
        ("h", 3, "一、梦的凝缩作用"),
        ("p", _BODY2),
    ])
    boundaries = _detect_boundaries(elems)
    # 版权信息区间（到"爱情心理学"前）无有效正文 → 被空区间过滤
    # 两个 h1 卷名（C 规则）+ 续编 h1（D 规则补充）共 3 卷
    assert len(boundaries) == 3, f"应检出 3 个卷边界，实际 {len(boundaries)}"
    assert boundaries[0].rule == "C"
    assert elems[boundaries[0].elem_index].text == "爱情心理学"
    assert elems[boundaries[1].elem_index].text == "达·芬奇的童年回忆"
    assert elems[boundaries[2].elem_index].text == "释梦（下）"
    assert boundaries[2].rule == "D"


def test_detect_philosophy_style_volumes():
    """哲学通史式：版权信息 h1 重复 5 次、章标题同为 h1 → B 类边界。"""
    items = [
        ("h", 1, "总目录"),
        ("p", "总目录页码略。"),
    ]
    for vol in range(3):  # 合成 3 卷即可验证
        items.append(("h", 1, "版权信息"))
        items.append(("p", f"版权所有侵权必究第{vol+1}卷出版信息定价发行。"))
        items.append(("h", 1, "导论"))
        items.append(("p", _BODY1))
        items.append(("h", 1, f"第一章 第{vol+1}卷的内容"))
        items.append(("h", 2, "第一节 概述"))
        items.append(("p", _BODY2))
        items.append(("h", 1, "主要参考文献"))
        items.append(("p", "参考文献列表若干条。"))
    elems = _stream(items)
    boundaries = _detect_boundaries(elems)
    assert len(boundaries) == 3, f"应检出 3 个卷边界，实际 {len(boundaries)}"
    assert all(b.rule == "B" for b in boundaries)
    assert all(b.title is None for b in boundaries)  # B 类无显式卷名


def test_detect_jung_style_single_volume():
    """荣格式：目录/部分 h1 + h2 作品，无编委会/版权页重复 → 单卷降级。"""
    elems = _stream([
        ("h", 1, "总目录"),
        ("p", "目录页。"),
        ("h", 1, "目录"),
        ("p", "本卷目录。"),
        ("h", 1, "第一部分 早期文章"),
        ("h", 2, "弗洛伊德的癔症理论"),
        ("p", _BODY1),
        ("h", 2, "梦的分析"),
        ("p", _BODY2),
        ("h", 1, "第二部分 精神分析理论"),
        ("h", 4, "第一版序言"),
        ("p", _BODY3),
        ("h", 4, "一、对早期假说的回顾"),
        ("p", _BODY4),
    ])
    boundaries = _detect_boundaries(elems)
    assert boundaries == [], "荣格式结构不应检出卷边界，应整书单卷落盘"


def test_front_back_matter_skipped():
    """编委会/编译说明/参考文献等行政页整节剔除，不进 contents。"""
    elems = _stream([
        ("h", 2, "《弗洛伊德文集》编委会"),
        ("p", "主编某某副主编某某编委名单。"),
        ("h", 2, "新版编译说明"),
        ("h", 3, "一、版本出处"),
        ("p", "版本出处说明文字。"),
        ("h", 2, "性学三论"),
        ("h", 3, "第一篇 性变态"),
        ("p", _BODY1),
        ("h", 2, "主要参考文献"),
        ("p", "文献条目一文献条目二文献条目三。"),
    ])
    chapters, front, back = _build_volume_tree(elems, 0, len(elems))
    assert front >= 2, "编委会/编译说明应计为剔除 front 节"
    assert back >= 1, "参考文献应计为剔除 back 节"
    all_text = " ".join(
        p for ch in chapters for sec in [ch] + ch.children for p in sec.paragraphs
    )
    assert "编委" not in all_text
    assert "文献条目" not in all_text
    assert "精神分析" in all_text


def test_leading_cip_debris_dropped():
    """B 类边界（版权页标题被排除在卷区间外）后残留的 CIP 无标题段落必须丢弃，
    不得生成"未命名章"兜底。"""
    elems = _stream([
        # 卷区间从版权页标题之后开始：先是无标题 CIP 碎片
        ("p", "书名：中国哲学通史（宋元卷）"),
        ("p", "ISBN 978-7-0000-0000-0 出版发行 印刷装订 定价。"),
        ("h", 1, "导论"),
        ("p", _BODY1),
        ("h", 1, "第一章 道学的先驱"),
        ("h", 2, "第一节 韩愈与佛教"),
        ("p", _BODY2),
    ])
    chapters, front, back = _build_volume_tree(elems, 0, len(elems))
    titles = [c.title for c in chapters]
    assert "未命名章" not in titles, f"卷首碎片不应生成未命名章，实际章列表：{titles}"
    assert titles == ["导论", "第一章 道学的先驱"]
    all_text = " ".join(
        p for ch in chapters for sec in [ch] + ch.children for p in sec.paragraphs
    )
    assert "ISBN" not in all_text and "书名：" not in all_text
    assert "精神分析" in all_text


# ==================== 阶段 1/2 端到端（合成 EPUB → 落盘） ====================

def _freud_like_epub(path: Path) -> Path:
    part1 = "".join([
        _h(1, "版权信息"), _p("版权所有侵权必究出版发行定价版次印次。"),
        _h(1, "爱情心理学"),
        _h(2, "《弗洛伊德文集》编委会"), _p("主编某某副主编某某编委名单。"),
        _h(2, "新版编译说明"), _h(3, "一、版本出处"), _p("版本出处说明。"),
        _h(2, "性学三论"),
        _h(3, "第一篇 性变态"),
        _h(4, "一、关于性对象的变异"),
        _p(_BODY1),
        _h(5, "（一）性变态"),
        _p(_BODY2),
        _h(2, "论自恋：导论"),
        _p(_BODY3),
    ])
    part2 = "".join([
        _h(1, "达·芬奇的童年回忆"),
        _h(2, "新版编译说明"), _p("编译说明同上。"),
        _h(2, "达·芬奇的童年回忆"),
        _h(3, "英文版编者导言"),
        _p(_BODY4),
        _h(3, "第一章"),
        _p(_BODY1),
        _h(2, "主要参考文献"),
        _p("参考文献条目若干。"),
    ])
    return _make_epub(path, "弗洛伊德文集（合成）", [part1, part2])


def test_stage1_extract_multivolume(tmp_path):
    """R-17：多卷合订本按卷独立落盘 extracted.md/.json，人工可读、含层级与溯源。"""
    epub_path = tmp_path / "freud" / "合成弗洛伊德.epub"
    epub_path.parent.mkdir(parents=True)
    _freud_like_epub(epub_path)

    out_dir = tmp_path / "extracted"
    results = extract_book(str(epub_path), out_dir=out_dir)

    assert len(results) == 2, f"应落盘 2 卷，实际 {len(results)}"
    for r in results:
        assert r.md_path.exists(), f"{r.md_path} 不存在"
        assert r.json_path.exists(), f"{r.json_path} 不存在"
        payload = json.loads(r.json_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2
        assert payload["source_document"]["domain"] == "freud"
        assert payload["source_document"]["content_hash"]
        assert payload["volume"]["volume_title"].startswith("卷")
        assert payload["outline_tree"], "outline_tree 不应为空"
        assert payload["contents"], "contents 不应为空"
        # contents 每项携带完整 path_ref
        for item in payload["contents"]:
            ref = item["path_ref"]
            assert ref["volume"]
            assert ref["chapter"]
            assert "section" in ref
            assert "paragraph_order_in_section" in item
        # md 人工可读：含卷名与锚点
        md = r.md_path.read_text(encoding="utf-8")
        assert md.startswith("# 卷")
        assert "{#ch-" in md

    # 卷 1 应包含 性学三论 章节，且编委会被剔除
    vol1 = json.loads(results[0].json_path.read_text(encoding="utf-8"))
    chapters1 = [c["title"] for c in vol1["outline_tree"]]
    assert "性学三论" in chapters1
    assert "《弗洛伊德文集》编委会" not in chapters1
    all_text1 = " ".join(i["text"] for i in vol1["contents"])
    assert "编委名单" not in all_text1
    # h5 更深小节路径拼接
    subsections = [
        i["path_ref"]["subsection"] for i in vol1["contents"]
        if i["path_ref"].get("subsection")
    ]
    assert any("·" in s for s in subsections), "h5（一）应拼接到 subsection 路径"


def test_stage1_single_volume_fallback(tmp_path):
    """无卷信号时整书单卷落盘（降级规则）。"""
    epub_path = tmp_path / "philosophy" / "合成大问题.epub"
    epub_path.parent.mkdir(parents=True)
    html = "".join([
        _h(1, "目录"), _p("目录页。"),
        _h(1, "导言：做哲学"),
        _h(2, "明确表达和论证"),
        _p(_BODY1),
        _h(1, "第一章 哲学问题"),
        _h(2, "哲学问题"),
        _p(_BODY2),
        _h(2, "参考书目与阅读建议"),
        _p("参考书目若干。"),
    ])
    _make_epub(epub_path, "大问题（合成）", [html])

    results = extract_book(str(epub_path), out_dir=tmp_path / "extracted")
    assert len(results) == 1, "单卷书应只落盘 1 个文件对"
    payload = json.loads(results[0].json_path.read_text(encoding="utf-8"))
    assert payload["volume"]["multi_volume"] is False
    chapters = [c["title"] for c in payload["outline_tree"]]
    assert "第一章 哲学问题" in chapters
    all_text = " ".join(i["text"] for i in payload["contents"])
    assert "参考书目" not in all_text


def test_stage2_chunks_traceability(tmp_path):
    """R-17：阶段 2 chunks.json 含 6 级溯源元数据 + 锚点 + 块序号。"""
    epub_path = tmp_path / "freud" / "合成弗洛伊德.epub"
    epub_path.parent.mkdir(parents=True)
    _freud_like_epub(epub_path)

    extracted_dir = tmp_path / "extracted"
    chunks_dir = tmp_path / "chunks"
    extract_book(str(epub_path), out_dir=extracted_dir)
    report = run_chunk_stage(extracted_dir=extracted_dir, out_dir=chunks_dir)

    assert len(report.volumes) == 2
    assert report.total_blocks > 0
    for vol in report.volumes:
        assert vol.error is None, vol.error
        data = json.loads(Path(vol.chunks_path).read_text(encoding="utf-8"))
        assert data["schema_version"] == 2
        assert data["blocks"], "blocks 不应为空"
        for b in data["blocks"]:
            ref = b["parent_ref"]
            # 6 级溯源：书/卷/章/节/小节 + 节内序号
            assert ref["book_title"]
            assert ref["volume_title"]
            assert ref["chapter_title"]
            assert "section_title" in ref
            assert "subsection_title" in ref
            assert isinstance(b["block_index_in_section"], int)
            assert b["block_id"] and b["block_type"] == "text"
            assert b["text"] and b["char_count"] == len(b["text"])
            assert b["source_anchor_in_md"].startswith("#ch-")
            assert b["soft_split_from_paragraphs"]
        # stats 占位（两级摘要阶段 3 生成）
        assert data["stats"]["total_chapter_summary_blocks"] == 0
        assert data["stats"]["total_section_summary_blocks"] == 0


def test_stage2_soft_split_long_paragraph(tmp_path):
    """超长段落软切：同一段落产生多块，soft_split_from_paragraphs 记录来源段号。"""
    epub_path = tmp_path / "philosophy" / "合成长段.epub"
    epub_path.parent.mkdir(parents=True)
    long_para = "这是一个非常长的段落。" * 120  # ~1080 字，超过 500 上限
    html = "".join([
        _h(1, "第一章 长段测试"),
        _h(2, "第一节"),
        _p(long_para),
        _p(_BODY2),
    ])
    _make_epub(epub_path, "长段（合成）", [html])

    extract_book(str(epub_path), out_dir=tmp_path / "extracted")
    report = run_chunk_stage(
        extracted_dir=tmp_path / "extracted", out_dir=tmp_path / "chunks",
    )
    blocks = json.loads(Path(report.volumes[0].chunks_path).read_text(encoding="utf-8"))["blocks"]
    long_blocks = [b for b in blocks if len(b["text"]) > 0]
    # 长段被切成多块，每块不超过 500 字（软切目标）
    oversize = [b for b in long_blocks if b["char_count"] > 520]
    assert not oversize, f"存在超长块：{[b['char_count'] for b in oversize]}"
    # 同一段落切出的多块共享来源段号
    para1_blocks = [b for b in blocks if b["soft_split_from_paragraphs"] == [1]]
    assert len(para1_blocks) >= 2


def test_run_extract_stage_scans_kb(tmp_path):
    """run_extract_stage 扫描知识库目录，按 domain 子目录落盘。"""
    kb = tmp_path / "knowledge_base"
    (kb / "Jung").mkdir(parents=True)
    html = "".join([
        _h(1, "第一部分 早期文章"),
        _h(2, "梦的分析"),
        _p(_BODY1),
    ])
    _make_epub(kb / "Jung" / "合成荣格.epub", "荣格文集（合成）", [html])

    report = run_extract_stage(kb_dir=kb, out_dir=tmp_path / "extracted")
    assert len(report.books) == 1
    assert report.total_volumes == 1
    assert (tmp_path / "extracted" / "jung").is_dir()
    json_files = list((tmp_path / "extracted" / "jung").glob("*.extracted.json"))
    assert len(json_files) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
