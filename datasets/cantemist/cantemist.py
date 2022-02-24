# Loading script for the Cantemist NER dataset. 
import datasets
logger = datasets.logging.get_logger(__name__)
_CITATION = """\
@inproceedings{miranda2020named,
  title={Named entity recognition, concept normalization and clinical coding: Overview of the cantemist track for cancer text mining in spanish, corpus, guidelines, methods and results},
  author={Miranda-Escalada, A and Farr{\'e}, E and Krallinger, M},
  booktitle={Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2020), CEUR Workshop Proceedings},
  year={2020}
}"""
_DESCRIPTION = """\
https://temu.bsc.es/cantemist/
"""
_URL = "https://huggingface.co/datasets/BSC-TeMU/cantemist-ner/resolve/main/"
# _URL = "./"
_TRAINING_FILE = "./train.conll"
_DEV_FILE = "./dev.conll"
_TEST_FILE = "./test.conll"

class CantemistNerConfig(datasets.BuilderConfig):
    """BuilderConfig for Cantemist Ner dataset"""
    def __init__(self, **kwargs):
        """BuilderConfig for CantemistNer.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CantemistNerConfig, self).__init__(**kwargs)

class CantemistNer(datasets.GeneratorBasedBuilder):
    """Cantemist Ner dataset."""
    BUILDER_CONFIGS = [
        CantemistNerConfig(
            name="CantemistNer", 
            version=datasets.Version("1.0.0"), 
            description="CantemistNer dataset"),
    ]
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-MORFOLOGIA_NEOPLASIA",
                                "I-MORFOLOGIA_NEOPLASIA",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://temu.bsc.es/cantemist/",
            citation=_CITATION,
        )
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_TRAINING_FILE}",
            "dev": f"{_DEV_FILE}",
            "test": f"{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]
    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            pos_tags = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # Cantemist tokens are tab separated
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[-1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
