## Mehr Corpus

Mehr_corpus is a Persian coreference resolution corpus and framework.

### How to Run

1. Open the full project in **PyCharm**.
2. Run `Mehr_Setting1_Create_fv` to create full feature vector CSV files.
3. Run `Mehr_Setting1_train` to perform 10-fold cross-validation with the best hyperparameters on the training documents.
4. Run `Mehr_Setting1_test_system` to evaluate the full chain system on the test set.  
   ⚠️ Note: This process may take several minutes to hours depending on your CPU.

---

## 📄 Reference

If you use **Mehr Corpus** in your research, please cite the following paper:

> Haji Mohammadi, H., Talebpour, A., Mahmoudi Aznaveh, A., & Yazdani, S. (2023).  
> **Mehr: A Persian Coreference Resolution Corpus.**  
> *Journal of AI and Data Mining, 11*(3), 407–416.  
> [https://jad.shahroodut.ac.ir/article_2897_1b4eef782cd783c3876250545dff45ed.pdf](https://jad.shahroodut.ac.ir/article_2897_1b4eef782cd783c3876250545dff45ed.pdf)

---

## 📘 Citation (BibTeX)

```bibtex
@article{hajimohammadi2023mehr,
  title={Mehr: A Persian Coreference Resolution Corpus},
  author={Haji Mohammadi, Hassan and Talebpour, Alireza and Mahmoudi Aznaveh, Ali and Yazdani, Saeed},
  journal={Journal of AI and Data Mining},
  volume={11},
  number={3},
  pages={407--416},
  year={2023},
  url={https://jad.shahroodut.ac.ir/article_2897_1b4eef782cd783c3876250545dff45ed.pdf}
}
