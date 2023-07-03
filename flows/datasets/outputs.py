from flows import utils
from flows.datasets import AbstractDataset
from flows.utils.general_helpers import read_outputs

log = utils.get_pylogger(__name__)


class OutputsDataset(AbstractDataset):
    def __init__(self, data=None, **kwargs):
        super().__init__(kwargs)

        self.data = data
        self.filter_failed = kwargs.get("filter_failed", True)

        if self.data is None:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # See the models.api_model._get_outputs_to_write() function for the data that is available for each datapoint
        return self.data[idx]

    def _load_data(self):
        self.data = read_outputs(self.params["data_dir"])

        if self.filter_failed:
            log.info("[Output DS] Filtering out the datapoints for which the prediction failed")
            self.data = [sample for sample in self.data if sample["error"] is None]

        if len(self.data) == 0:
            log.warning("[Output DS] No predictions were loaded from %s", self.params["data_dir"])
        else:
            log.info(
                "[Output DS] Loaded the predictions for %d datapoints from %s", len(self.data), self.params["data_dir"]
            )
