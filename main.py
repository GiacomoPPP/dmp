from RunType import RunType

from ScaleInvarianceAnalayzer import ScaleInvarianceAnalayzer

from TrainingRunner import TrainingRunner

from DirectionListAnalyzer import DirectionListAnalyzer

from TargetDistributionAnalyzer import TargetDistributionAnalyzer

from DmpConfig import DmpConfig

import MplStyle # Necessary to globally set matplotlib style

config = DmpConfig()

match config.run_type[0]:
    case RunType.TRAIN:
        trainingRunner = TrainingRunner()
        trainingRunner()
    case RunType.ANALYZE_DIRECTIONS:
        directionListAnalyzer = DirectionListAnalyzer()
        directionListAnalyzer()
    case RunType.ANALYZE_TARGET_DISTRIBUTION:
        target_ditribution_analyzer = TargetDistributionAnalyzer()
        target_ditribution_analyzer()
    case RunType.ANALYZE_SCALE_INVARIANCE:
        scale_invariance_analyzer = ScaleInvarianceAnalayzer()
        scale_invariance_analyzer()