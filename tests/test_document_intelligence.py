from app.document_intelligence import detect_subjects, extract_grade


def test_extract_grade_from_transliterated_schoolbook_filename():
    assert extract_grade("russkij-jazyk-9-2-chast-kibireva.pdf") == 9


def test_extract_grade_keeps_explicit_class_pattern():
    assert extract_grade("Русский язык. 10 класс") == 10


def test_extract_grade_does_not_treat_arbitrary_number_as_grade():
    assert extract_grade("история отечественной войны 1812 года") is None


def test_detect_subject_from_transliterated_filename():
    detected = detect_subjects(
        "4e684ce46f_geografija_9_klass_e_a_tamozhnjaja_2022_g.pdf"
    )

    assert detected[0]["subject"] == "geography"


def test_detect_subjects_from_final_textbook_queries():
    assert detect_subjects("Какими средствами выражается сравнение?")[0]["subject"] == "russian_language"
    assert detect_subjects("Почему выпадает чёрный осадок сульфида меди?")[0]["subject"] == "chemistry"
    assert detect_subjects("Как автомобильный транспорт загрязняет окружающую среду?")[0]["subject"] == "geography"
