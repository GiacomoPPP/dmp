from DatasetParser import DatasetParser
from RunType import RunType

from ScaleInvarianceAnalayzer import ScaleInvarianceAnalayzer

from SingleTrainingRunner import SingleTrainingRunner
from MultipleTrainingRunner import MultipleTrainingRunner

from DirectionListAnalyzer import DirectionListAnalyzer

from TargetDistributionAnalyzer import TargetDistributionAnalyzer

from DmpConfig import DmpConfig

import MplStyle # Necessary to globally set matplotlib style

config = DmpConfig()

match config.run_type[0]:
    case RunType.TRAIN_SINGLE:
        trainingRunner = SingleTrainingRunner()
        trainingRunner()
    case RunType.TRAIN_MULTIPLE:
        trainingRunner = MultipleTrainingRunner()
        trainingRunner()
    case RunType.PARSE_DATASET:
        datasetParser = DatasetParser()
        datasetParser()
    case RunType.ANALYZE_DIRECTIONS:
        directionListAnalyzer = DirectionListAnalyzer()
        directionListAnalyzer()
    case RunType.ANALYZE_TARGET_DISTRIBUTION:
        target_ditribution_analyzer = TargetDistributionAnalyzer()
        target_ditribution_analyzer()
    case RunType.ANALYZE_SCALE_INVARIANCE:
        scale_invariance_analyzer = ScaleInvarianceAnalayzer()
        scale_invariance_analyzer()