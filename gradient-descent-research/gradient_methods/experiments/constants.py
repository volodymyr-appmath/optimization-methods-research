from ..methods import CGD, Broyden, DFP, NewtonMethod

method_name_mapping = {
    "cgd": CGD,
    "broyden": Broyden,
    "dfp": DFP,
    "newton": NewtonMethod,
}

en_to_ua_mapping = {
    "cgd": "МСГ",
    "broyden": "Метод Бройдена",
    "dfp": "Метод ДФП",
    "newton": "Метод Ньютона",
}
