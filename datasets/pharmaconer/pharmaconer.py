# Loading script for the PharmaCoNER dataset. 
import datasets
logger = datasets.logging.get_logger(__name__)
_CITATION = """\
@inproceedings{gonzalez-agirre-etal-2019-pharmaconer,
    title = "{P}harma{C}o{NER}: Pharmacological Substances, Compounds and proteins Named Entity Recognition track",
    author = "Gonzalez-Agirre, Aitor  and
      Marimon, Montserrat  and
      Intxaurrondo, Ander  and
      Rabal, Obdulia  and
      Villegas, Marta  and
      Krallinger, Martin",
    booktitle = "Proceedings of The 5th Workshop on BioNLP Open Shared Tasks",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-5701",
    doi = "10.18653/v1/D19-5701",
    pages = "1--10",
    abstract = "One of the biomedical entity types of relevance for medicine or biosciences are chemical compounds and drugs. The correct detection these entities is critical for other text mining applications building on them, such as adverse drug-reaction detection, medication-related fake news or drug-target extraction. Although a significant effort was made to detect mentions of drugs/chemicals in English texts, so far only very limited attempts were made to recognize them in medical documents in other languages. Taking into account the growing amount of medical publications and clinical records written in Spanish, we have organized the first shared task on detecting drug and chemical entities in Spanish medical documents. Additionally, we included a clinical concept-indexing sub-track asking teams to return SNOMED-CT identifiers related to drugs/chemicals for a collection of documents. For this task, named PharmaCoNER, we generated annotation guidelines together with a corpus of 1,000 manually annotated clinical case studies. A total of 22 teams participated in the sub-track 1, (77 system runs), and 7 teams in the sub-track 2 (19 system runs). Top scoring teams used sophisticated deep learning approaches yielding very competitive results with F-measures above 0.91. These results indicate that there is a real interest in promoting biomedical text mining efforts beyond English. We foresee that the PharmaCoNER annotation guidelines, corpus and participant systems will foster the development of new resources for clinical and biomedical text mining systems of Spanish medical data.",
}
"""
_DESCRIPTION = """\
https://temu.bsc.es/pharmaconer/
"""
_URL = "https://huggingface.co/datasets/BSC-TeMU/pharmaconer/resolve/main/"
# _URL = "./"
_TRAINING_FILE = "./train.conll"
_DEV_FILE = "./dev.conll"
_TEST_FILE = "./test.conll"

class PharmaCoNERConfig(datasets.BuilderConfig):
    """BuilderConfig for PharmaCoNER dataset"""
    def __init__(self, **kwargs):
        """BuilderConfig for PharmaCoNER.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PharmaCoNERConfig, self).__init__(**kwargs)
class PharmaCoNER(datasets.GeneratorBasedBuilder):
    """PharmaCoNER dataset."""
    BUILDER_CONFIGS = [
        PharmaCoNERConfig(
            name="PharmaCoNER", 
            version=datasets.Version("1.0.0"), 
            description="PharmaCoNER dataset"),
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
                                "B-NO_NORMALIZABLES",
                                "B-NORMALIZABLES",
                                "B-PROTEINAS",
                                "B-UNCLEAR",
                                "I-NO_NORMALIZABLES",
                                "I-NORMALIZABLES",
                                "I-PROTEINAS",
                                "I-UNCLEAR",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://temu.bsc.es/pharmaconer/",
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
                    # PharmaCoNER tokens are tab separated
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[-1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
