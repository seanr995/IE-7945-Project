import json, os
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(ROOT, 'sprint4_results'))
R = json.load(open('run_summary.json')); O = R['onet']; S = R['stability']; G = R['gap']; LG = R['ledger']
D = pd.read_csv('hierarchy_diagnostics.csv'); L = pd.read_csv('leaf_topics_labeled.csv'); E = pd.read_csv('leaf_evidence.csv')
H = pd.read_csv('human_review_template.csv')
HUMAN_COLS = ['label_correct', 'coherent', 'boundary_ok', 'over_merged', 'under_merged']
assert len(H) == 80 and H.topic_id.nunique() == 80 and H[HUMAN_COLS].notna().all().all(), 'human review incomplete'
_y = lambda c, v: int((H[c].astype(str).str.strip().str.upper() == v).sum())
HR = dict(n=len(H), label_no=_y('label_correct', 'NO'), coh_yes=_y('coherent', 'YES'), coh_no=_y('coherent', 'NO'), bnd_no=_y('boundary_ok', 'NO'),
          over=_y('over_merged', 'YES'), under=_y('under_merged', 'YES'), ren=int(H.suggested_name.notna().sum()))
assert HR['coh_yes'] + HR['coh_no'] == 80
OVER = ', '.join(H[H.over_merged.str.upper() == 'YES'].topic_id); UNDER = ', '.join(H[H.under_merged.str.upper() == 'YES'].topic_id)
search = D[D.stage == 'search']
a20 = search[(search.linkage == 'average') & (search.level2_count == 20) & (search.level1_count == 8)].iloc[0]
w20 = search[(search.linkage == 'ward') & (search.level2_count == 20) & (search.level1_count == 8)].iloc[0]
n_avg_pass = int(search[search.linkage == 'average'].passes_gates.sum()); n_w_pass = int(search[search.linkage == 'ward'].passes_gates.sum())
assert (n_avg_pass, n_w_pass) == (0, 9)
N1, N2 = R['level1'], R['level2']
assert len(L) == 80 and L.topic_size.sum() == 3015

NAVY = RGBColor(0x1F, 0x3A, 0x5F); TEAL = RGBColor(0x1F, 0x8A, 0x84); AMBER = RGBColor(0xD9, 0x8E, 0x1F); GREY = RGBColor(0x5B, 0x65, 0x70)
LIGHT = RGBColor(0xEE, 0xF2, 0xF6); WHITE = RGBColor(255, 255, 255); RED = RGBColor(0xB2, 0x3A, 0x48); DARK = RGBColor(0x22, 0x2B, 0x33)
PURPLE = RGBColor(0x6B, 0x4C, 0x9A); MIDBLUE = RGBColor(0x3C, 0x6E, 0xA8)

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
NOTES = []


def txt(slide, x, y, w, h, text, size=16, color=DARK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.02)
    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        r = p.add_run(); r.text = ln; r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic; r.font.color.rgb = color; r.font.name = 'Calibri'
    return tb


def box(slide, x, y, w, h, text, fill=NAVY, size=16, color=WHITE, bold=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h)); s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line; s.line.width = Pt(1.5)
    s.shadow.inherit = False
    tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.08); tf.margin_top = tf.margin_bottom = Inches(0.04)
    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        r = p.add_run(); r.text = ln; r.font.size = Pt(size if i == 0 else min(size - 3, 14)); r.font.bold = bold if i == 0 else False; r.font.color.rgb = color; r.font.name = 'Calibri'
    return s


def arrow(slide, x, y, w=0.3, h=0.35, color=GREY):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h)); a.fill.solid(); a.fill.fore_color.rgb = color; a.line.fill.background(); return a


def darrow(slide, x, y, w=0.35, h=0.3, color=GREY):
    a = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(x), Inches(y), Inches(w), Inches(h)); a.fill.solid(); a.fill.fore_color.rgb = color; a.line.fill.background(); return a


def tile(slide, x, y, w, h, big, label, fill=LIGHT, bigcolor=NAVY, bigsize=34, labsize=13):
    box(slide, x, y, w, h, '', fill=fill, line=None)
    txt(slide, x, y + 0.08, w, h * 0.55, big, size=bigsize, color=bigcolor, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(slide, x + 0.05, y + h * 0.58, w - 0.1, h * 0.4, label, size=labsize, color=GREY, align=PP_ALIGN.CENTER)


def new(title, subtitle=None, notes=''):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12)); bar.fill.solid(); bar.fill.fore_color.rgb = TEAL; bar.line.fill.background()
    txt(s, 0.5, 0.3, 12.3, 0.8, title, size=30, color=NAVY, bold=True)
    if subtitle: txt(s, 0.5, 1.05, 12.3, 0.45, subtitle, size=16, color=GREY)
    txt(s, 0.5, 7.08, 9, 0.3, 'Sprint 4 | Level 1 = work domains | Level 2 = activity families | Level 3 = leaf task topics', size=10, color=GREY)
    s.notes_slide.notes_text_frame.text = notes; NOTES.append((title, notes))
    return s


def legend(s, y=6.7):
    box(s, 0.5, y, 0.28, 0.22, '', fill=TEAL, shape=MSO_SHAPE.RECTANGLE); txt(s, 0.82, y - 0.05, 3.2, 0.3, 'Deterministic computation', size=11, color=GREY)
    box(s, 3.6, y, 0.28, 0.22, '', fill=PURPLE, shape=MSO_SHAPE.RECTANGLE); txt(s, 3.92, y - 0.05, 3.2, 0.3, 'Gemini semantic reasoning', size=11, color=GREY)
    box(s, 6.7, y, 0.28, 0.22, '', fill=AMBER, shape=MSO_SHAPE.RECTANGLE); txt(s, 7.02, y - 0.05, 3.2, 0.3, 'O*NET external reference', size=11, color=GREY)
    box(s, 9.9, y, 0.28, 0.22, '', fill=RED, shape=MSO_SHAPE.RECTANGLE); txt(s, 10.22, y - 0.05, 3.0, 0.3, 'Human review (complete)', size=11, color=GREY)


# ------------------------------------------------------------------ Slide 1
s = new('Sprint 4: From 80 Leaf Topics to a Hierarchical Taxonomy', 'Frozen job-posting topics, named and organised into a labelled hierarchy, then checked against O*NET',
        'Open with the one-line story. We did not re-cluster anything: the 80 leaf topics from Sprint 3 stay exactly as they were. Sprint 4 adds names, two parent levels, and an outside check against O*NET. '
        f'The final structure is {N1} work domains over {N2} activity families over 80 leaves. Point out the colour code: teal is deterministic computation, purple is Gemini, amber is O*NET, red is human review, which is complete (80 of 80 reviewed).')
steps = [('3,015', 'validated tasks', NAVY), ('80', 'frozen Leiden leaf topics', TEAL), ('80', 'labelled leaves (Gemini)', PURPLE),
         (str(N2), 'Level-2 activity families', TEAL), (str(N1), 'Level-1 work domains', TEAL), ('O*NET', 'external evaluation', AMBER)]
x = 0.4
for i, (big, lab, c) in enumerate(steps):
    box(s, x, 2.3, 1.85, 1.9, [big, lab], fill=c, size=30)
    if i < 5: arrow(s, x + 1.88, 3.08, 0.28, 0.35)
    x += 2.14
box(s, 0.5, 4.75, 12.3, 1.2, [f'Final structure: {N1} Level-1 work domains  >  {N2} Level-2 activity families  >  80 Level-3 leaf topics  >  3,015 tasks',
                              'Task memberships are frozen: Sprint 4 never moved a task between leaves.'], fill=LIGHT, color=NAVY, size=20, bold=True)
txt(s, 0.5, 6.1, 12.3, 0.4, f'Human hierarchy review: COMPLETE, {HR["n"]} of 80 topics reviewed; {HR["over"]} over-merged, {HR["under"]} under-merged flagged.', size=15, color=RED, bold=True)
legend(s)

# ------------------------------------------------------------------ Slide 2
s = new('What Entered Sprint 4', 'Inputs were fixed before any labelling or hierarchy work began',
        'Stress that Sprint 4 starts from Sprint 3\'s final model: Sentence-T5 embeddings clustered with Leiden at 80 topics. We froze it so that anything we change later can be attributed to naming or hierarchy, not to re-clustering. '
        'We verified byte-for-byte that all 3,015 task-to-leaf assignments are identical in the Sprint 4 output. Say the key sentence clearly: O*NET did not create our topics. It is an outside reference used only to evaluate.')
tile(s, 0.6, 1.8, 3.8, 1.7, '3,015', 'validated job-posting tasks')
tile(s, 4.75, 1.8, 3.8, 1.7, '80', 'frozen Sentence-T5 + Leiden leaf topics', bigcolor=TEAL)
tile(s, 8.9, 1.8, 3.8, 1.7, '0', 'tasks moved or re-clustered in Sprint 4', bigcolor=RED)
box(s, 0.6, 3.95, 5.9, 1.9, ['Empirical taxonomy', 'Built from current job postings: tasks > Sentence-T5 > Leiden leaves > hierarchy'], fill=TEAL, size=20)
box(s, 6.8, 3.95, 5.9, 1.9, ['O*NET reference data', 'Used only as an external evaluation / reference frame (18,838 task statements)'], fill=AMBER, size=20)
box(s, 2.2, 6.0, 8.9, 0.6, 'O*NET did not create our topics.', fill=NAVY, size=24)
legend(s)

# ------------------------------------------------------------------ Slide 3
s = new('Naming the 80 Leaf Topics', 'Every leaf is described from evidence, labelled by Gemini, and validated locally',
        'Walk the pipeline left to right. For each leaf we computed the centroid, picked the 10 core tasks closest to it, the 5 boundary tasks furthest from it, and c-TF-IDF keywords. Gemini then got a compact slice of that evidence (7 core, 3 boundary, 8 keywords) '
        'and returned a structured label. Python validated every record against the schema and word limits; Gemini never wrote a file. The three examples are real rows from the results. One topic, LT_048, was flagged by Gemini as mixed, which is why review matters.')
ev = [('Centroid', TEAL), ('Core + boundary tasks', TEAL), ('c-TF-IDF keywords', TEAL), ('Gemini semantic label', PURPLE), ('Structured validation', TEAL)]
x = 0.45
for i, (t, c) in enumerate(ev):
    box(s, x, 1.65, 2.2, 0.85, t, fill=c, size=15)
    if i < 4: arrow(s, x + 2.22, 1.92, 0.25, 0.3)
    x += 2.5
for i, tid in enumerate(['LT_001', 'LT_002', 'LT_028']):
    r = L[L.topic_id == tid].iloc[0]; e = E[E.topic_id == tid].iloc[0]; core = json.loads(e.core_tasks)
    x0 = 0.5 + i * 4.15
    box(s, x0, 2.85, 3.95, 0.7, f'{tid}  |  {r.topic_name}', fill=NAVY, size=16)
    box(s, x0, 3.55, 3.95, 2.7, '', fill=LIGHT, shape=MSO_SHAPE.RECTANGLE)
    txt(s, x0 + 0.1, 3.62, 3.75, 0.3, f'{int(r.topic_size)} tasks | label confidence {r.label_confidence:.2f}', size=12, color=GREY)
    txt(s, x0 + 0.1, 3.95, 3.75, 0.3, 'Representative tasks', size=13, color=NAVY, bold=True)
    txt(s, x0 + 0.1, 4.25, 3.75, 0.9, ['- ' + core[0], '- ' + core[1]], size=14)
    txt(s, x0 + 0.1, 5.1, 3.75, 0.3, 'Description', size=13, color=NAVY, bold=True)
    txt(s, x0 + 0.1, 5.4, 3.75, 1.05, r.description, size=12)
txt(s, 0.5, 6.3, 12.3, 0.35, 'Gemini mixed-topic flags: 1 of 80 (LT_048). Unresolved labels: 0.', size=12, color=GREY)
legend(s)

# ------------------------------------------------------------------ Slide 4
s = new('Building the Hierarchy', 'Both linkage methods ran through the same quantitative gates; the numbers chose Ward',
        'Level-3 leaves roll up into Level-2 activity families and Level-1 work domains, both cut from one dendrogram so they are nested. We tested average linkage and Ward linkage on the 80 leaf centroids, 9 candidates each, with identical gates and the same score. '
        f'No average-linkage candidate passed the gates: at 20 and 8, one domain swallowed {a20.L1_dominant_share*100:.0f} percent of all tasks. All nine Ward candidates passed; Ward 20/8 had the highest score of {w20.score:.3f}. So the selection was evidence-based, not hand-picked. '
        'Then one Gemini critic round reviewed the structure and moved a single leaf. Use the small real example on the right: Strategic Planning and Financial Management with its two families.')
lv = [(f'{N1} Level-1 work domains', NAVY), (f'{N2} Level-2 activity families', MIDBLUE), ('80 leaf topics', TEAL), ('3,015 tasks', GREY)]
y = 1.7
for i, (t, c) in enumerate(lv):
    box(s, 0.5, y, 4.0, 0.7, t, fill=c, size=18)
    if i < 3: darrow(s, 2.3, y + 0.71, 0.35, 0.22)
    y += 0.93
cd = CategoryChartData(); cd.categories = ['Average linkage', 'Ward linkage']; cd.add_series('Largest Level-1 domain, share of tasks (20 L2 / 8 L1)', [round(float(a20.L1_dominant_share), 3), round(float(w20.L1_dominant_share), 3)])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(4.8), Inches(1.6), Inches(3.9), Inches(3.3), cd); ch = gf.chart
ch.has_legend = False; ch.has_title = True; ch.chart_title.text_frame.text = 'Largest Level-1 domain (share of tasks)'; ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
pl = ch.plots[0]; pl.gap_width = 60; pl.has_data_labels = True; pl.data_labels.number_format = '0.0%'; pl.data_labels.number_format_is_linked = False; pl.data_labels.font.size = Pt(14); pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
ch.value_axis.maximum_scale = 1.0; ch.value_axis.visible = False; ch.value_axis.has_major_gridlines = False; ch.category_axis.tick_labels.font.size = Pt(13)
pts = pl.series[0].points; pts[0].format.fill.solid(); pts[0].format.fill.fore_color.rgb = RED; pts[1].format.fill.solid(); pts[1].format.fill.fore_color.rgb = TEAL
box(s, 4.8, 5.0, 1.9, 1.0, [f'{n_avg_pass}/9', 'average candidates pass gates'], fill=RED, size=22)
box(s, 6.9, 5.0, 1.8, 1.0, [f'{n_w_pass}/9', 'Ward candidates pass gates'], fill=TEAL, size=22)
txt(s, 4.8, 6.05, 3.9, 0.6, f'Average: {int(a20.L1_max_tasks):,} tasks in one domain, {int(a20.L2_tiny_parents)} tiny L2 parents. Ward: max {int(w20.L1_max_tasks)}, score {w20.score:.3f} (selected).', size=11, color=GREY)
d4 = L[L.level1_id == 'D1_04']; fams = d4.groupby('level2_name').topic_name.apply(list)
box(s, 9.0, 1.65, 3.85, 0.75, f'L1: {d4.level1_name.iloc[0]}', fill=NAVY, size=14)
yy = 2.65
for fam, leaves in fams.items():
    box(s, 9.2, yy, 3.65, 0.65, f'L2: {fam}', fill=MIDBLUE, size=12)
    txt(s, 9.4, yy + 0.68, 3.45, 1.2, ['- ' + n for n in leaves[:3]], size=12, color=DARK)
    yy += 1.95
legend(s)

# ------------------------------------------------------------------ Slide 5
s = new('O*NET Evaluation', 'Retrieval is deterministic; Gemini only judges what was already retrieved',
        'Explain the order of operations. For every leaf centroid we computed cosine similarity to all 18,838 O*NET task statements and took the exact top 10. That is plain linear algebra. Only after that did Gemini see the top candidates and judge the semantic match. '
        'Gemini did not select the nearest O*NET tasks. Metrics: coverage is 0.975 because 78 of 80 topics matched meaningfully. Purity, 0.1935, is concept agreement, not accuracy: it is the share of tasks in a topic whose nearest O*NET task falls in the dominant O*NET concept of that topic. It is low in absolute terms because members of one topic map to many different O*NET activities. Top-10 nearest-neighbour agreement is 0.6225. '
        f'Gemini judged {O["alignment_counts"]["STRONG_MATCH"]} strong, {O["alignment_counts"]["PARTIAL_MATCH"]} partial and {O["alignment_counts"]["NO_MATCH"]} no-match topics.')
fl = [('Sentence-T5 leaf centroid', TEAL), ('Cosine vs all 18,838 O*NET tasks', TEAL), ('Exact Top-10 (deterministic)', TEAL), ('Gemini semantic evaluation', PURPLE)]
x = 0.5
for i, (t, c) in enumerate(fl):
    box(s, x, 1.7, 2.8, 1.0, t, fill=c, size=16)
    if i < 3: arrow(s, x + 2.83, 2.05, 0.3, 0.32)
    x += 3.15
tile(s, 0.5, 3.1, 2.3, 1.6, f'{R["onet_tasks"]:,}', 'O*NET 31.0 task statements', bigsize=28, bigcolor=AMBER)
tile(s, 3.0, 3.1, 2.3, 1.6, f'{O["coverage_meaningful_alignment"]:.3f}', 'coverage (78 of 80 topics)', bigcolor=TEAL)
tile(s, 5.5, 3.1, 2.3, 1.6, f'{O["purity_mean"]:.4f}', f'purity = concept agreement, not accuracy (median {O["purity_median"]:.3f})', bigcolor=TEAL)
tile(s, 8.0, 3.1, 2.3, 1.6, f'{O["nn_agreement_top10_mean"]:.4f}', f'NN agreement, top-10 (exact top-1 {O["nn_agreement_top1_exact_mean"]:.3f})', bigcolor=TEAL)
tile(s, 10.5, 3.1, 2.3, 1.6, f'{O["top1_mean"]:.3f}', 'mean top-1 cosine', bigcolor=TEAL)
ac = O['alignment_counts']
box(s, 0.5, 5.0, 3.9, 0.95, [f'{ac["STRONG_MATCH"]} strong', 'Gemini judgment'], fill=PURPLE, size=22)
box(s, 4.7, 5.0, 3.9, 0.95, [f'{ac["PARTIAL_MATCH"]} partial', 'Gemini judgment'], fill=PURPLE, size=22)
box(s, 8.9, 5.0, 3.9, 0.95, [f'{ac["NO_MATCH"]} no match', 'Gemini judgment (LT_060, LT_076)'], fill=PURPLE, size=22)
box(s, 0.5, 6.1, 12.3, 0.5, 'Gemini did NOT select the nearest O*NET tasks.', fill=NAVY, size=20)
legend(s, 6.78)

# ------------------------------------------------------------------ Slide 6
s = new('Beyond Basic Sprint 4: Robustness and Coverage Gaps', 'Three additions that test the hierarchy and show where evidence is thin',
        f'A: We perturbed the hierarchy five times, with seeds 101 to 505, each time keeping a random 80 percent of the tasks inside every leaf, recomputing centroids and re-clustering. Persistence is how often a leaf kept its parent. '
        f'Mean persistence is {S["mean_level1_persistence"]:.4g} at Level 1 and {S["mean_level2_persistence"]:.4g} at Level 2, ARI against the final hierarchy {S["level1_ari_vs_final_mean"]:.2f} and {S["level2_ari_vs_final_mean"]:.2f}. That is moderate, not perfect, and we report it honestly. '
        'B: The gap analysis uses quantile cut-offs from our own data. Possible emerging work means job-posting evidence has limited alignment with O*NET; it does not prove O*NET is outdated. C: 80 evidence cards, one per topic, are what reviewers will read.')
txt(s, 0.5, 1.55, 6.2, 0.4, 'A. Hierarchy stability', size=20, color=NAVY, bold=True)
txt(s, 0.5, 1.95, 6.2, 0.7, '5 deterministic perturbation trials | 80% within-topic task subsampling | recomputed centroids | parent persistence + ARI', size=13, color=GREY)
tile(s, 0.5, 2.8, 1.45, 1.45, f'{S["mean_level1_persistence"]:.4g}', 'L1 persistence (mean)', bigsize=26, labsize=11, bigcolor=TEAL)
tile(s, 2.1, 2.8, 1.45, 1.45, f'{S["mean_level2_persistence"]:.4g}', 'L2 persistence (mean)', bigsize=26, labsize=11, bigcolor=TEAL)
tile(s, 3.7, 2.8, 1.45, 1.45, f'{S["level1_ari_vs_final_mean"]:.2f}', 'L1 ARI vs final', bigsize=26, labsize=11, bigcolor=TEAL)
tile(s, 5.3, 2.8, 1.45, 1.45, f'{S["level2_ari_vs_final_mean"]:.2f}', 'L2 ARI vs final', bigsize=26, labsize=11, bigcolor=TEAL)
txt(s, 0.5, 4.35, 6.3, 0.7, f'Pairwise ARI: L1 {S["level1_ari_pairwise_mean"]:.2f}, L2 {S["level2_ari_pairwise_mean"]:.2f}. Leaves with persistence < 0.60: L1 {S["leaves_level1_persistence_below_0.60"]}, L2 {S["leaves_level2_persistence_below_0.60"]}. Moderate stability; validation only, hierarchy not changed.', size=12, color=GREY)
txt(s, 7.1, 1.55, 5.8, 0.4, 'B. O*NET coverage-gap analysis', size=20, color=NAVY, bold=True)
gc = G['counts']
cd = CategoryChartData(); cd.categories = ['WELL_COVERED', 'PARTIALLY_COVERED', 'POSSIBLE_EMERGING_WORK']; cd.add_series('Topics', [gc['WELL_COVERED'], gc['PARTIALLY_COVERED'], gc['POSSIBLE_EMERGING_WORK']])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(7.1), Inches(1.95), Inches(5.8), Inches(2.4), cd); ch = gf.chart; ch.has_legend = False; ch.has_title = False
pl = ch.plots[0]; pl.gap_width = 50; pl.has_data_labels = True; pl.data_labels.font.size = Pt(15); pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
ch.value_axis.visible = False; ch.value_axis.has_major_gridlines = False; ch.value_axis.maximum_scale = 55; ch.category_axis.tick_labels.font.size = Pt(12); ch.category_axis.reverse_order = True
for i, c in enumerate([TEAL, AMBER, RED]):
    pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = c
txt(s, 7.1, 4.4, 5.8, 0.9, 'Possible emerging work = current job-posting task evidence has limited alignment with the reference taxonomy. It does NOT prove O*NET is outdated.', size=12, color=GREY, italic=True)
box(s, 0.5, 5.4, 12.3, 1.0, ['C. Explainability: 80 topic-level evidence cards', 'description, representative + boundary tasks, keywords, hierarchy path, O*NET matches, persistence'], fill=LIGHT, color=NAVY, size=20)
legend(s)

# ------------------------------------------------------------------ Slide 7
s = new('Human Validation', f'Completed manual review of all {HR["n"]} leaf topics: useful, with real over- and under-merging',
        f'Human review is complete: all {HR["n"]} topics were judged on five questions and the CSV has no blanks. Labels: {HR["n"]-HR["label_no"]} accepted, {HR["label_no"]} need a rename (LT_015, LT_071, LT_076); suggested names are reported, not applied. '
        f'Representative tasks: {HR["coh_yes"]} topics coherent, {HR["coh_no"]} mixed. Boundary tasks: {HR["bnd_no"]} topics have boundary tasks that do not fit, which is the weakest area. '
        f'Over-merged: {HR["over"]} topics ({OVER}); these are grab-bags glued by a word like lead or drive. Under-merged: {HR["under"]} topics ({UNDER}); they form near-duplicate pairs such as LT_004 and LT_061, LT_026 and LT_042, LT_021 and LT_052. '
        'We did not change any memberships or names: merges and renames are future work. This is a review by our team, not ground truth.')
box(s, 0.5, 1.6, 12.3, 0.6, f'HUMAN REVIEW COMPLETE: {HR["n"]} of {HR["n"]} topics reviewed', fill=TEAL, size=20)
tl = [(f'{HR["n"]-HR["label_no"]}/{HR["n"]}', 'labels accepted', TEAL), (f'{HR["coh_yes"]}/{HR["n"]}', 'coherent topics', TEAL), (f'{HR["bnd_no"]}', 'topics with boundary issues', AMBER),
      (f'{HR["over"]}', 'over-merged topics', RED), (f'{HR["under"]}', 'under-merged topics', RED), (f'{HR["ren"]}', 'suggested renames', PURPLE)]
for i, (b, l_, c) in enumerate(tl):
    tile(s, 0.5 + i * 2.07, 2.45, 1.95, 1.5, b, l_, bigcolor=c, bigsize=30, labsize=12)
box(s, 0.5, 4.2, 6.0, 1.0, ['Over-merged (mixed content)', OVER], fill=RED, size=15)
box(s, 6.8, 4.2, 6.0, 1.0, ['Under-merged (near-duplicate topics)', UNDER], fill=RED, size=15)
txt(s, 0.5, 5.35, 12.3, 1.2, [f'Suggested renames (reported, not applied): LT_015 > Lead Projects and Business Initiatives; LT_071 > Build Reusable Assets and Company-wide Processes; LT_076 > Lead Organizational Change and Expansion.',
    'Unresolved: no merges, splits or renames applied; 3,015 task memberships and 80 leaves unchanged. Review by project team; not ground truth.'], size=14, color=DARK)
legend(s)

# ------------------------------------------------------------------ Slide 8
s = new('Sprint 4 Outcome / Sprint 5 Handoff', 'Actual saved outputs; human review complete, issues documented',
        'Summarise the deliverables: 80 labelled leaves, 20 families, 8 domains, 3,015 frozen task memberships, an O*NET evaluation, a stability test and 80 evidence cards. Gemini made 10 real calls with no repairs or retries, and Sprint 4 made no Claude calls. '
        'Close with the chain at the bottom and the caveat: this is an interpretable, human-auditable structure, not ground truth. Human review is complete: it accepted 77 of 80 labels and 74 of 80 topics as coherent, but found 5 over-merged and 12 under-merged topics. Sprint 5 should apply the 3 renames and decide on merging the near-duplicate pairs.')
t8 = [('80', 'labelled leaves'), (str(N2), 'Level-2 families'), (str(N1), 'Level-1 domains'), ('3,015', 'frozen tasks')]
for i, (b, l_) in enumerate(t8):
    tile(s, 0.5 + i * 3.1, 1.6, 2.9, 1.5, b, l_, bigcolor=TEAL if i < 3 else NAVY)
items = [f'O*NET evaluation: coverage {O["coverage_meaningful_alignment"]:.3f}, purity {O["purity_mean"]:.4f}, NN agreement {O["nn_agreement_top10_mean"]:.4f}',
         f'Hierarchy stability: persistence L1 {S["mean_level1_persistence"]:.4g} / L2 {S["mean_level2_persistence"]:.4g}; ARI vs final {S["level1_ari_vs_final_mean"]:.2f} / {S["level2_ari_vs_final_mean"]:.2f}',
         f'Gap analysis: {gc["WELL_COVERED"]} well / {gc["PARTIALLY_COVERED"]} partial / {gc["POSSIBLE_EMERGING_WORK"]} possible emerging',
         f'80 explainability cards | {LG["gemini_real_calls"]} Gemini calls | 0 Claude calls | 0 unresolved',
         f'Human review: COMPLETE ({HR["n"]}/80): {HR["over"]} over-merged, {HR["under"]} under-merged, {HR["label_no"]} label corrections suggested']
for i, it in enumerate(items):
    txt(s, 0.7, 3.3 + i * 0.42, 12, 0.4, '- ' + it, size=15, color=RED if i == 4 else DARK, bold=(i == 4))
ch8 = ['Current job-posting evidence', 'Interpretable task taxonomy', 'Hierarchical organisation', 'External O*NET evaluation', 'Human-auditable structure']
cs = [TEAL, TEAL, TEAL, AMBER, RED]
x = 0.45
for i, t in enumerate(ch8):
    box(s, x, 5.65, 2.2, 0.9, t, fill=cs[i], size=14)
    if i < 4: arrow(s, x + 2.22, 5.93, 0.25, 0.3)
    x += 2.5
txt(s, 0.5, 6.62, 12.3, 0.3, 'The hierarchy is an auditable organisation of the evidence, not ground truth.', size=12, color=GREY, italic=True)

OUT = os.path.join(ROOT, 'Sprint4_Labeling_Hierarchy_Presentation.pptx')
SUB = os.path.join(ROOT, 'SPRINT4_FINAL_SUBMISSION', 'Sprint4_Labeling_Hierarchy_Presentation.pptx')
prs.save(SUB)
try:
    prs.save(OUT)
except PermissionError:
    print('WARNING: root pptx is locked (open in PowerPoint?); saved only to', SUB)
with open(os.path.join(ROOT, 'Sprint4_Presentation_Speaker_Notes.md'), 'w', encoding='utf-8') as f:
    f.write('# Sprint 4 Presentation - Speaker Notes\n\nAll figures are real values from `sprint4_results/` (see `Sprint4_Presentation_Key_Metrics.md`). Human review is complete (80/80, values from human_review_template.csv); it is a team review, not ground truth.\n\n')
    for i, (t, n) in enumerate(NOTES, 1):
        f.write(f'## Slide {i} - {t}\n\n{n}\n\n')
print('saved', OUT)
