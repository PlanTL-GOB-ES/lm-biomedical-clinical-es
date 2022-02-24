# Biomedical and clinical language model for Spanish 🏥
Repository of the biomedical and clinical pretrained language models for Spanish. 
For more details about the corpus, the pretraining and the evaluation, read our [preprint](https://arxiv.org/abs/2109.03570) "_Carrino, C. P., Armengol-Estapé, J., Gutiérrez-Fandiño, A., Llop-Palao, J., Pàmies, M., Gonzalez-Agirre, A., & Villegas, M. (2021). Biomedical and Clinical Language Models for Spanish: On the Benefits of Domain-Specific Pretraining in a Mid-Resource Scenario._"

## Corpora 📃
The training corpora are composed of several biomedical corpora in Spanish, collected from publicly available corpora and crawlers, and a real-world clinical corpus collected from more than 278K clinical documents and notes. To obtain a high-quality training corpus while retaining the idiosyncrasies of the clinical language, a cleaning pipeline has been applied only to the biomedical corpora, keeping the clinical corpus uncleaned. Essentially, the cleaning operations used are:

- data parsing in different formats
  - sentence splitting
  - language detection
  - filtering of ill-formed sentences 
  - deduplication of repetitive contents
  - keep the original document boundaries

Then, the biomedical corpora are concatenated and further global deduplication among the biomedical corpora have been applied.
The results are a *clinical corpus* of about 91M tokens and a *biomedical corpus* of about 963M tokens. 
Eventually, the clinical corpus is concatenated to the cleaned biomedical corpus resulting in a medium-size biomedical-clinical corpus for Spanish composed of more than 1B tokens. The table below shows some basic statistics of the individual cleaned corpora:

    
| Name                                                                                    | No. tokens  | Description                                                                                                                                                                                                                                          |
|-----------------------------------------------------------------------------------------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Medical crawler](https://zenodo.org/record/4561970)                                    | 745,705,946 | Crawler of more than 3,000 URLs belonging to Spanish biomedical and health domains.                                                                                                                                                                                 |
| Clinical cases misc.                                                                    | 102,855,267 | A miscellany of medical content, essentially clinical cases. Note that a clinical case report is a scientific publication where medical practitioners share patient cases and it is different from a clinical note or document.                                                                                                                                                                                 |
| Clinical notes/documents                                                                | 91,250,080 | Collection of more than 278K clinical documents, including discharge reports, clinical course notes and X-ray reports, for a total of 91M tokens.                                                                                                                                                                                 |
| [Scielo](https://github.com/PlanTL-GOB-ES/SciELO-Spain-Crawler)                        | 60,007,289  | Publications written in Spanish crawled from the Spanish SciELO server in 2017.                                                                                                                                       |
| [BARR2_background](https://temu.bsc.es/BARR2/downloads/background_set.raw_text.tar.bz2) | 24,516,442  | Biomedical Abbreviation Recognition and Resolution (BARR2) containing Spanish clinical case study sections from a variety of clinical disciplines.                                                                                       |
| Wikipedia_life_sciences                                                                 | 13,890,501  | Wikipedia articles crawled 04/01/2021 with the [Wikipedia API python library](https://pypi.org/project/Wikipedia-API/) starting from the "Ciencias\_de\_la\_vida" category up to a maximum of 5 subcategories. Multiple links to the same articles are then discarded to avoid repeating content.                                                                                                                                                                    |
| Patents                                                                                 | 13,463,387  | Google Patent in Medical Domain for Spain (Spanish). The accepted codes (Medical Domain) for Json files of patents are: "A61B", "A61C","A61F", "A61H", "A61K", "A61L","A61M", "A61B", "A61P".                                                        |
| [EMEA](http://opus.nlpl.eu/download.php?f=EMEA/v3/moses/en-es.txt.zip)                  | 5,377,448   | Spanish-side documents extracted from parallel corpora made out of PDF documents from the European Medicines Agency.                                                                                                                            |
| [mespen_Medline](https://zenodo.org/record/3562536#.YTt1fH2xXbR)                        | 4,166,077   | Spanish-side articles extracted from a collection of Spanish-English parallel corpus consisting of biomedical scientific literature.  The collection of parallel resources are aggregated from the MedlinePlus source. |
| PubMed                                                                                  | 1,858,966   | Open-access articles from the PubMed repository crawled in 2017.                                                                                                                                                                                  
## Models 🤖
We released our best models under the Apache License 2.0 to encourage the development of Spanish NLP applications in the biomedical and clinical domains. You can access our best biomedical and biomedical-clinical models in the HuggingFace models hub at the following links:
 - [biomedical](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-es): the bio-52k-SWM model in our paper
 - [biomedical-clinical](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es): the bio-cli-52k model in our paper

## Evaluation ✅
The model has been evaluated on the Named Entity Recognition (NER) using the following datasets [PharmaCoNER](https://zenodo.org/record/4270158), [CANTEMIST](https://zenodo.org/record/3978041#.YTt5qH2xXbQ) and ICTUSnet, and are compared against the [mBERT](https://huggingface.co/bert-base-multilingual-cased) and [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) models:

| Task/models | roberta-base-biomedical-clinical-es | roberta-base-biomedical-es | mBERT                   | BETO                    |
|---------------------------|----------------------------|-------------------------------|-------------------------------|-------------------------|
|                           |F1 \| Precision \| Recall     |F1 \| Precision \| Recall|F1 \| Precision \| Recall|F1 \| Precision \| Recall|
| PharmaCoNER               | **90.04** \| 88.92 \| 91.18    | 89.48 \| 87.85 \| 91.18 |87.46 \| 86.50 \| 88.46 | 88.18 \| 87.12 \| 89.28 |
| CANTEMIST                 | 83.34 \| 81.48 \| 85.30    | **83.87** \| 81.70 \| 86.17 | 82.61 \| 81.12 \| 84.15 | 82.42 \| 80.91 \| 84.00 |
| ICTUSnet                  | 88.08 \| 84.92 \| 91.50    | **88.12** \| 85.56 \| 90.83 |86.75 \| 83.53 \| 90.23 | 85.95 \| 83.10 \| 89.02 |

## Usage example ⚗️

```python

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-biomedical-es")

model = AutoModelForMaskedLM.from_pretrained("PlanTL-GOB-ES/roberta-base-biomedical-es")

from transformers import pipeline

unmasker = pipeline('fill-mask', model="PlanTL-GOB-ES/roberta-base-biomedical-es")

unmasker("El único antecedente personal a reseñar era la <mask> arterial.")

```

```

# Output

[
  {
    "sequence": " El único antecedente personal a reseñar era la hipertensión arterial.",
    "score": 0.9855039715766907,
    "token": 3529,
    "token_str": " hipertensión"
  }
]

 
```

## Cite 📣
If you use our models, please cite our latest preprint:

```bibtex

@misc{carrino2021biomedical,
      title={Biomedical and Clinical Language Models for Spanish: On the Benefits of Domain-Specific Pretraining in a Mid-Resource Scenario}, 
      author={Casimiro Pio Carrino and Jordi Armengol-Estapé and Asier Gutiérrez-Fandiño and Joan Llop-Palao and Marc Pàmies and Aitor Gonzalez-Agirre and Marta Villegas},
      year={2021},
      eprint={2109.03570},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```

If you use our Medical Crawler corpus, please cite the preprint:

```bibtex

@misc{carrino2021spanish,
      title={Spanish Biomedical Crawled Corpus: A Large, Diverse Dataset for Spanish Biomedical Language Models}, 
      author={Casimiro Pio Carrino and Jordi Armengol-Estapé and Ona de Gibert Bonet and Asier Gutiérrez-Fandiño and Aitor Gonzalez-Agirre and Martin Krallinger and Marta Villegas},
      year={2021},
      eprint={2109.07765},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```

## How to run ️▶️

First, clone the project.

Then, you will need to set up a virtual environment and activate it.

Install all the required python packages in the virtual environment running the following line from the project main folder: `pip install -r requirements.txt`.

Finally, run the following command on the project main folder: `bash ner.sh [MODEL_NAME] [DATABASE_NAME] [SEED]`, where

- [MODEL_NAME] is the name of the pretrained model you want to use.
- [DATABASE_NAME] can be chosen from "cantemist" and "pharmaconer", depending on the dataset you want to use.
- [SEED] is the seed you want to use. This allows to reproduce the same results.
