# Polish-Kashubian Translator Project

The project consists of a Polish-Kashubian translator based on a seq2seq model, enhanced by an attention mechanism.

The model was trained and evaluated using the Polish-Kashubian Parallel Translation Corpus (Olewniczak et al., 2024).

## Installation 
1. Clone the repository
    ```bash
   git clone git@github.com:mariah881/polkash_translator_project.git
    ```
2. Navigate to the project directory
    ```bash
    cd polkash_translator_project
    ```
3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
4. Download the model from HuggingFace using the link below:
    [Link text](https://huggingface.co/hmaria/polkash_translator_project)
4. Run the command below to load the model and perform inference:
    ```bash
    cd python
    python inference.py
    ```



## References

Olewniczak, S., Nowak, M., Szweda, F., Źęgota, J., Kulpiński, K., Wrzosek, M., Grzybowski, J., & Czepiel, K. (2024). Polish-Kashubian parallel translation corpus (Version 1.0, 1–) [Dataset]. Gdańsk University of Technology. https://doi.org/10.34808/t930-fs97



