# PA4 - Transformers for Amazon Scenario classification

This project explores transformer-based models for scenario classification using the Amazon Massive Scenario Dataset. The implementation includes:

-Baseline Model: Fine-tuning a pre-trained BERT model.
-Custom Fine-Tuning: Enhancing the baseline using advanced techniques (LLRD and Frequent Validation).
-Contrastive Learning: Applying SupContrast and SimCLR for learning better representations.

The loss and accuracy for each model are saved in the "results" folder. The plots for train and validation accuracy are in the "plots" folder."

To run:
main.py --task baseline/custom/supcon/supcon --use_sim --batch-size 32 --n-epochs 10 --do-train

