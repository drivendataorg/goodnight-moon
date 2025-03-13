[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[<img src='https://s3.amazonaws.com/drivendata-public-assets/mit-RER-banner-image.jpg'>]({https://www.drivendata.org/competitions/298/literacy-screening/})

# Goodnight Moon, Hello Early Literacy Screening

## Goal of the Competition

Literacy—the ability to read, write, and comprehend language—is a fundamental skill that underlies personal development, academic success, career opportunities, and active participation in society. Many children in the United States need more support with their language skills. In order to provide effective early literacy intervention, teachers must be able to reliably identify the students who need support. Currently, teachers across the U.S. are tasked with administering and scoring literacy screeners, which are written or verbal tests that are manually scored following detailed rubrics. Manual scoring methods not only take time, but they may be unreliable, producing different results depending on who is scoring the test and how thoroughly they were trained.

__In this challenge, solvers implemented machine learning models to score audio recordings from literacy screener exercises completed by students in kindergarten through 3rd grade. The models can help teachers quickly and reliably identify children in need of early literacy intervention.__

This competition also included a bonus for explainability write-ups that described methods to determine where in the audio stream error(s) occur and provide insight into the model decision-making rationale.

## What's in this Repository

This repository contains code from winning competitors in the [Goodnight Moon, Hello Early Literacy Screening]({https://www.drivendata.org/competitions/298/literacy-screening/}) DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Public Score | Private Score | Summary of Model
--- | --- | ---   | ---   | ---
1   | sheep | 0.2011 | 0.2042 | Fine-tuned OpenAI’s Whisper model with a custom loss function. For each word in expected_text, its token was inputed into the decoder and computed the binary cross-entropy (BCE) loss between the token’s logit and its corresponding score.
2   | dylanliu | 0.2088 | 0.2098 | Combined various multimodal classification models, including a Whisper-medium (only the encoder part) as the speech feature extraction model and bge-large as the text feature extraction model. Used a cv-based weight searching script to automatically search for weights.
3   | vecxoz | 0.2156 | 0.2137 | Used 4 modalities of the data: original audio, audio generated from expected text (by SpeechT5), original expected text, and text transcribed from original audio (by Whisper finetuned on positive examples). Built a multimodal classifier which uses the encoder from the finetuned Whisper model as audio feature extractor and Deberta-v3 as text feature extractor.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Winners Blog Post: [{winners-blog-post-title}]({winners-blog-post-url})**

**Benchmark Blog Post: [Goodnight Moon, Hello Early Literacy Screening Benchmark](https://drivendata.co/blog/literacy-screening-benchmark)**
