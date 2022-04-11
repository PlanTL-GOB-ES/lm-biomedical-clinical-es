# Biomedical and clinical language model for Spanish 🏥
Official repository of the Barcelona Supercomputing Center (BSC) biomedical and clinical pretrained language models for Spanish.

## Models 🤖
We released our best models under the Apache License 2.0 to encourage the development of Spanish NLP applications in the biomedical and clinical domains. 

### Pretrained models:
We trained two models and make them available on HuggingFace' models hub at the following links:

 - [bsc-bio-es](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-es): pretrained language model suitable for biomedical and clinical documents
 - [bsc-bio-ehr-es](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es): pretrained language model suitable for EHR documents and clinical notes

### Fine-tuned NER models:
We fine-tuned our models for the Named Entity Recognition (NER) task and make the best models available on HuggingFace' models hub at the following links:

 - [bsc-bio-ehr-es-cantemist](https://huggingface.co/PlanTL-GOB-ES/bsc-bio-ehr-es-cantemist): NER model for tumor morphology on clinical cases
 - [bsc-bio-ehr-es-pharmaconer](https://huggingface.co/PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer): NER model for substances, compounds and proteins on clinical cases

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
| [Medical crawler](https://zenodo.org/record/4561970)                                    | 903,558,136 | Crawler of more than 3,000 URLs belonging to Spanish biomedical and health domains.                                                                                                                                                                                 |
| Clinical cases misc.                                                                    | 102,855,267 | A miscellany of medical content, essentially clinical cases. Note that a clinical case report is a scientific publication where medical practitioners share patient cases and it is different from a clinical note or document.                                                                                                                                                                                 |
| EHR documents                                                                | 95,267,20 | Collection of more than 278K clinical documents, including discharge reports, clinical course notes and X-ray reports, for a total of 91M tokens.                                                                                                                                                                                 |
| [Scielo](https://zenodo.org/record/2541681#.YlP1DshBwio)                        | 60,007,289  | Publications written in Spanish crawled from the Spanish SciELO server in 2017.                                                                                                                                       |
| [BARR2_background](https://temu.bsc.es/BARR2/downloads/background_set.raw_text.tar.bz2) | 24,516,442  | Biomedical Abbreviation Recognition and Resolution (BARR2) containing Spanish clinical case study sections from a variety of clinical disciplines.                                                                                       |
| Wikipedia_life_sciences                                                                 | 13,890,501  | Wikipedia articles crawled 04/01/2021 with the [Wikipedia API python library](https://pypi.org/project/Wikipedia-API/) starting from the "Ciencias\_de\_la\_vida" category up to a maximum of 5 subcategories. Multiple links to the same articles are then discarded to avoid repeating content.                                                                                                                                                                    |
| Patents                                                                                 | 13,463,387  | Google Patent in Medical Domain for Spain (Spanish). The accepted codes (Medical Domain) for Json files of patents are: "A61B", "A61C","A61F", "A61H", "A61K", "A61L","A61M", "A61B", "A61P".                                                        |
| [EMEA](http://opus.nlpl.eu/download.php?f=EMEA/v3/moses/en-es.txt.zip)                  | 5,377,448   | Spanish-side documents extracted from parallel corpora made out of PDF documents from the European Medicines Agency.                                                                                                                            |
| [mespen_Medline](https://zenodo.org/record/3562536#.YlP1UshBwio)                        | 4,166,077   | Spanish-side articles extracted from a collection of Spanish-English parallel corpus consisting of biomedical scientific literature.  The collection of parallel resources are aggregated from the MedlinePlus source. |
| PubMed                                                                                  | 1,858,966   | Open-access articles from the PubMed repository crawled in 2017.                                                                                                                                                                   

## Evaluation ✅
The models have been fine-tuned on the the Named Entity Recognition (NER) tasks using three clinical NER datasets, such as, [PharmaCoNER](https://zenodo.org/record/4270158), [CANTEMIST](https://zenodo.org/record/3978041#.YTt5qH2xXbQ) and ICTUSnet. We addressed the NER task as a token classification problem using a standard linear layer along with the BIO tagging schema. We compared our models with the general-domain Spanish [roberta-base-bne](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne), the general-domain multilingual model that supports Spanish [mBERT](https://huggingface.co/bert-base-multilingual-cased), the domain-specific English model [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2), and three domain-specific models based on continual pre-training, [mBERT-Galén](https://ieeexplore.ieee.org/document/9430499), [XLM-R-Galén](https://ieeexplore.ieee.org/document/9430499) and [BETO-Galén](https://ieeexplore.ieee.org/document/9430499). The table below shows the F1 scores obtained:

| Tasks/Models | bsc-bio-es   | bsc-bio-ehr-es | XLM-R-Galén        | BETO-Galén   | mBERT-Galén  | mBERT        | BioBERT      | roberta-base-bne |
|--------------|--------------|----------------|--------------------|--------------|--------------|--------------|--------------|------------------|
| PharmaCoNER  | 0.8907 | **0.8913**   | 0.8754       | 0.8537 | 0.8594 | 0.8671 | 0.8545 | 0.8474     |
| CANTEMIST    | 0.8220  | **0.8340**    | 0.8078 | 0.8153 | 0.8168 | 0.8116 | 0.8070  | 0.7875     |
| ICTUSnet     | 0.8727 | **0.8756**   | 0.8716       | 0.8498 | 0.8509 | 0.8631 | 0.8521 | 0.8677 |

## NER Usage example ⚗️

```python

from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer")

model = AutoModelForTokenClassification.from_pretrained("PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer")

```

## How to run ️▶️

First, clone the project.

Then, you will need to set up a virtual environment and activate it.

Install all the required python packages in the virtual environment running the following line from the project main folder: `pip install -r requirements.txt`.

Finally, run the following command on the project main folder: `bash ner.sh [MODEL_NAME] [DATABASE_NAME] [SEED]`, where

- [MODEL_NAME] is the name of the pretrained model you want to use.
- [DATABASE_NAME] can be chosen from "cantemist" and "pharmaconer", depending on the dataset you want to use.
- [SEED] is the seed you want to use. This allows to reproduce the same results.


## Cite 📣
To be announced soon!

---

## Funding
This work was funded by the Spanish State Secretariat for Digitalization and Artificial Intelligence (SEDIA) within the framework of the Plan-TL.

## Disclaimer

The models published in this repository are intended for a generalist purpose and are available to third parties. These models may have bias and/or any other undesirable distortions.

When third parties, deploy or provide systems and/or services to other parties using any of these models (or using systems based on these models) or become users of the models, they should note that it is their responsibility to mitigate the risks arising from their use and, in any event, to comply with applicable regulations, including regulations regarding the use of artificial intelligence.

In no event shall the owner of the models (SEDIA – State Secretariat for digitalization and artificial intelligence) nor the creator (BSC – Barcelona Supercomputing Center) be liable for any results arising from the use made by third parties of these models.


Los modelos publicados en este repositorio tienen una finalidad generalista y están a disposición de terceros. Estos modelos pueden tener sesgos y/u otro tipo de distorsiones indeseables.

Cuando terceros desplieguen o proporcionen sistemas y/o servicios a otras partes usando alguno de estos modelos (o utilizando sistemas basados en estos modelos) o se conviertan en usuarios de los modelos, deben tener en cuenta que es su responsabilidad mitigar los riesgos derivados de su uso y, en todo caso, cumplir con la normativa aplicable, incluyendo la normativa en materia de uso de inteligencia artificial.

En ningún caso el propietario de los modelos (SEDIA – Secretaría de Estado de Digitalización e Inteligencia Artificial) ni el creador (BSC – Barcelona Supercomputing Center) serán responsables de los resultados derivados del uso que hagan terceros de estos modelos.
