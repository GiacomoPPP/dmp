from typing import Dict
from DmpConfig import DmpConfig
from DmpDataset import DmpDataset
from TrainingRunner import TrainingRunner

config = DmpConfig()

class MultipleTrainingRunner(TrainingRunner):

    def __call__(self):

        self._check_name_availability(config.model_name)

        test_error_list: Dict[DmpDataset, float] = {}

        for dataset in DmpDataset:
            _, test_error = self._train_and_assess_on_dataset(dataset)
            test_error_list[dataset] = test_error

        self.modelAssessment.write_results(test_error_list)


    def _check_name_availability(self, model_name: str):
        model_name_list: list[str] = self.modelAssessment.get_saved_model_names()
        if model_name in model_name_list:
            raise Exception(f"{model_name} was already used as model name")