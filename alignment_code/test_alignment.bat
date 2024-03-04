..\venv\Scripts\activate.bat
python .\unsup_align.py
    --model_src "..\alignment\unaligned_models\es_bible_model.vec"
    --model_tgt "..\alignment\unaligned_models\fr_bible_model.vec"
    --lexicon "..\alignment\unaligned_models\es_to_fr.lex"
    --output_src "..\alignment\aligned_models\es_aligned_model.vec"
    --output_tgt "..\alignment\aligned_models\fr_aligned_model.vec"
    --nmax 4776