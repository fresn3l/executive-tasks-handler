"""Run the desk end to end on one combined log."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lte_analyzer.flags import classify, cluster_areas
from lte_analyzer.model import ModelResult, fit_throughput
from lte_analyzer.relate import relationship_text
from lte_analyzer.report import build_report


@dataclass
class Analysis:
    frame: pd.DataFrame
    areas: pd.DataFrame
    relation: str
    model: ModelResult
    report_md: str


def analyze(frame: pd.DataFrame) -> Analysis:
    flagged = classify(frame)
    areas = cluster_areas(flagged)
    relation = relationship_text(flagged)
    model = fit_throughput(flagged)
    report = build_report(flagged, areas, relation, model)
    return Analysis(frame=flagged, areas=areas, relation=relation, model=model, report_md=report)
