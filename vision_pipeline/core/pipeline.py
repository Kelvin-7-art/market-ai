import logging
from core.result import AnalysisResult

log = logging.getLogger(__name__)


class VisionPipeline:
    def __init__(self):
        self.analyzers = []

    def register(self, analyzer):
        self.analyzers.append(analyzer)
        return self

    def run(self, image, context=None):
        context = context if context is not None else {}
        results = {}
        for a in self.analyzers:
            try:
                r = a.analyze(image, context)
                results[a.name] = r
                if r.boxes:
                    context.setdefault("boxes", []).extend(r.boxes)
            except Exception as e:
                log.exception("Analyzer %s failed", a.name)
                results[a.name] = AnalysisResult(a.name, error=str(e))
        return results
