"""
Thematic stock groupings sourced from @Speculator_io's image tables on X.
Each theme is a list of tickers; overlaps are intentional for rotation detection.

Each theme is also tagged with which Macro Rotation Group(s) it belongs to.
This enables the bridge: when the Macro View tab signals "Speculative Risk-On
group is firing", the Speculative Themes tab automatically highlights themes
in that bucket.
"""

# =============================================================================
# THEME → TICKER DICTIONARY
# =============================================================================

THEMES = {
    "AI_Supply_Chain": [
        "AXTI", "LWLG", "AAOI", "AEHR", "SNDK", "ICHR", "OPTX", "LITE", "CRDO", "MRVL",
        "INTC", "VIAV", "BE", "FORM", "WDC", "MU", "VRT", "ASX", "CAMT", "COHR", "GLW",
        "FN", "APH", "KLAC", "LRCX", "ASML", "TSM", "MPWR", "ON", "ETN", "ARM", "AOSL",
        "TSEM", "ALAB", "AVGO", "AMD", "ANET", "NVDA", "CIEN", "MTSI", "ACMR", "VICR",
        "PSIX", "STX", "TER", "ONTO", "LASR", "POET", "KEYS"
    ],
    "Nvidia_Key_Suppliers": [
        "ARM", "TSM", "INTC", "MU", "SNDK", "WDC", "ASX", "AMKR", "CAMT", "KLAC", "LRCX",
        "ASML", "KEYS", "COHR", "GLW", "FN", "LITE", "APH", "DELL", "SMCI", "JBL", "FLEX",
        "ETN", "VRT", "STM", "ADI", "MPWR", "NVTS", "ON", "CRWV", "NBIS", "NOK", "SNPS"
    ],
    "AI_Compute_Infrastructure": [
        "GOOGL", "AMZN", "MSFT", "META", "BABA", "ORCL", "NBIS", "CRWV",
        "IREN", "HUT", "WULF", "APLD", "CORZ", "RIOT", "CIFR",
        "MARA", "CLSK", "BTDR", "HIVE", "VRT", "DELL", "SMCI"
    ],
    "AI_Memory_Supercycle": [
        "MU", "SNDK", "WDC", "STX", "PSTG", "NTAP", "RMBS", "MRVL", "SIMO", "LRCX", "AMAT"
    ],
    "Critical_Minerals_RareEarths": [
        "MP", "IDR", "USAR", "TMQ", "CRML", "TMC", "TMRC", "UAMY", "NB", "UUUU",
        "PPTA", "IPX", "SCCO", "TGB", "ERO", "HBM", "FCX", "LEU", "UEC",
        "CCJ", "URG", "DNN", "NXE", "EU", "UROY",
        "ALB", "LAC", "SGML", "IONR", "SQM", "RIO", "ATLX", "SLI", "ABAT", "NMG", "WWR"
    ],
    "Nuclear_Energy": [
        "OKLO", "SMR", "NNE", "GEV", "BWXT", "LEU", "ASPI", "LTBR", "CCJ", "UEC",
        "BHP", "UUUU", "DNN", "NXE", "CW", "FLR", "BEPC", "CEG", "VST",
        "TLN", "PEG", "BEP", "NLR", "URA", "URNM", "NUKZ"
    ],
    "Space_Economy": [
        "RKLB", "RDW", "LUNR", "SPCE",
        "ASTS", "GSAT", "SATS", "IRDM", "GILT", "VSAT", "PL", "SATL",
        "BKSY", "SPIR", "MTRN", "HXL", "ATI", "GLW", "RTX", "LMT", "KTOS",
        "LHX", "NOC", "BA", "TDY", "APH", "PH", "AME", "HEI",
        "TDG", "AIR"
    ],
    "Drones_eVTOL": [
        "KTOS", "RCAT", "AVAV", "ESLT", "AIRO", "UMAC", "DPRO",
        "ONDS", "TDY", "HON", "PDYN", "ACHR", "JOBY", "EH",
        "LMT", "NOC", "BA", "RTX", "LHX", "TRMB", "UAVS", "TXT",
        "EADSY"
    ],
    "Photonics": [
        "AXTI", "IQE", "LWLG", "WOLF", "VECO",
        "LRCX", "AMAT", "ONTO", "MTSI", "HIMX", "STM", "LASR", "TSEM",
        "FN", "SANM", "UMC", "GFS", "AEHR", "FORM", "COHU", "TER", "VIAV", "ASX",
        "AMKR", "LITE", "COHR", "POET", "AAOI", "OPTX", "CRDO", "MRVL",
        "SMTC", "CIEN", "NOK", "GLW", "ALAB", "ANET", "AVGO"
    ],
    "Market_NextGen_Speculative_Leaders": [
        "SNDK", "BE", "STX", "LITE", "WDC", "CIEN", "NOK", "FIX", "VRT", "MRVL",
        "ASX", "ARM", "MU", "UI", "GLW", "NBIS", "AAOI", "MXL", "ICHR", "VIAV",
        "FSLY", "POWL", "AMPX", "VICR", "FORM", "AGX", "DOCN", "ADEA", "GNRC",
        "PL", "VSAT", "AXTI", "LWLG", "AEHR", "SATL", "OPTX", "SPIR", "NVTS",
        "UAMY", "USAR", "BKSY", "WULF", "FCEL", "CRML", "IBRX", "ERAS", "CRCL", "INOD",
        "FTAI", "HSAI", "NVMI", "OSCR", "ZETA", "ABCL"
    ],
    "Robotics_Quantum_Batteries_Multi": [
        "RR", "SERV", "ARBE", "MBOT", "PRCT", "PDYN", "QBTS", "QUBT", "RGTI", "ARQQ",
        "IONQ", "EOSE", "KULR", "ENVX", "MVST", "FLNC", "SLDP"
    ],
    "EV_Batteries": [
        "ENVX", "FLNC", "EOSE", "KULR", "SLDP", "QS", "MVST", "AMPX",
        "PLUG", "FCEL", "LAC", "ALB"
    ],
    "Hydrogen_Fuel_Cells": [
        "BE", "PLUG", "FCEL", "BLDP", "NRGV"
    ],
    "Biotech_Health": [
        "GRAL", "GH", "INSM", "TMDX", "WGS", "IDXX", "IRMD", "HIMS", "VEEV", "ALHC",
        "PODD", "TEM", "QURE", "EDIT", "RXRX", "SWTX", "AXSM", "CORT", "MIRM", "ADMA",
        "TGTX", "MRNA", "ARGX", "NTRA", "LLY", "PLNT", "LTH", "ULS", "IBRX", "ERAS",
        "ABCL", "OSCR"
    ],
    "FinTech": [
        "HOOD", "COIN", "XYZ", "FOUR", "ETOR", "BULL", "LMND", "ROOT", "SEZL", "DAVE",
        "PYPL", "IBKR", "SOFI", "UPST", "RKT", "TOST", "PRCH", "AFRM", "PAYO"
    ],
    "Insurance_InsureTech": [
        "ROOT", "PRCH", "LMND", "SLQT", "PRU", "MET", "LNC", "AFL", "CLOV", "OSCR",
        "GOCO", "ALHC", "HIPO", "ALL", "TRV", "CB", "HIG", "WRB", "PGR", "TIPT",
        "ERIE", "CINF", "AIG", "ACGL", "CNO", "AXS", "PLMR"
    ],
    "Copper_Supercycle": [
        "TMQ", "TGB", "SCCO", "FCX", "ERO", "CMCL", "COPX", "HBM"
    ],
    "Cybersecurity_FounderLed": [
        "NET", "ZS", "OKTA", "CRWD", "S", "CHKP", "AKAM"
    ],
    "Solar_Energy": [
        "ENLT", "BE", "GEV", "PWR", "EME", "GNRC", "FSLR"
    ],
    "Semiconductors": [
        "NVDA", "AVGO", "AMD", "TXN", "ADI", "MPWR", "QCOM", "AMAT", "LRCX",
        "KLAC", "TER", "ASML", "TSM", "INTC", "MU", "SNDK", "WDC", "STX"
    ],
    "Humanoid_Robots": [
        "FIGR", "SERV", "RR", "ARBE", "MBOT", "PRCT", "PDYN", "SYM", "AMZN"
    ],
    "Aerospace_Defense": [
        "RTX", "LMT", "KTOS", "LHX", "NOC", "BA", "AIR", "TXT", "EADSY",
        "HII", "TDY", "ESLT", "AVAV"
    ],
    "Power_Infrastructure": [
        "PWR", "GEV", "EME", "GNRC", "ETN", "VRT", "BE", "CEG", "VST"
    ],
    "Autonomous_Vehicles": [
        "TSLA", "AEVA", "HSAI", "TRMB"
    ],
    "Most_Shorted": [
        "QUBT", "ARQQ", "KULR", "ONDS", "CLSK", "MARA", "SERV", "WULF", "BTBT",
        "RGTI", "RCAT", "CIFR", "APPS", "QS", "QBTS", "RDW", "IONQ", "SMR", "HIMS",
        "LTBR", "OKLO", "NNE", "RIVN", "JOBY", "RKLB", "ACHR", "LUNR", "ASTS"
    ],
    "Crypto": [
        "COIN", "MARA", "CLSK", "RIOT", "HUT", "IREN", "WULF", "BTDR", "HIVE", "BTBT"
    ],
    "Software": [
        "SNOW", "DDOG", "APP", "TTD", "NOW", "WIX", "ZETA", "DOCN"
    ],
}

# =============================================================================
# MACRO ROTATION GROUP TAGGING
# Each theme tagged with which Macro View rotation group(s) it maps to.
# A theme can belong to multiple groups (rotation overlaps are a feature).
# =============================================================================

THEME_TO_MACRO_GROUPS = {
    # Speculative Risk-On — growth, software, biotech, retail, regional banks, crypto
    "AI_Supply_Chain":              ["Speculative_Risk_On"],
    "Nvidia_Key_Suppliers":         ["Speculative_Risk_On"],
    "AI_Compute_Infrastructure":    ["Speculative_Risk_On"],
    "AI_Memory_Supercycle":         ["Speculative_Risk_On"],
    "Photonics":                    ["Speculative_Risk_On"],
    "Market_NextGen_Speculative_Leaders": ["Speculative_Risk_On"],
    "Biotech_Health":               ["Speculative_Risk_On"],
    "FinTech":                      ["Speculative_Risk_On"],
    "Crypto":                       ["Speculative_Risk_On"],
    "Software":                     ["Speculative_Risk_On"],
    "Semiconductors":               ["Speculative_Risk_On"],
    "Most_Shorted":                 ["Speculative_Risk_On"],

    # Cyclical Expansion — industrials, transports, builders, capital goods
    "Aerospace_Defense":            ["Cyclical_Expansion", "Idiosyncratic"],
    "Power_Infrastructure":         ["Cyclical_Expansion"],
    "Drones_eVTOL":                 ["Cyclical_Expansion", "Idiosyncratic"],
    "Space_Economy":                ["Cyclical_Expansion", "Idiosyncratic"],
    "Humanoid_Robots":              ["Cyclical_Expansion"],
    "Autonomous_Vehicles":          ["Cyclical_Expansion"],
    "Robotics_Quantum_Batteries_Multi": ["Cyclical_Expansion", "Speculative_Risk_On"],

    # Commodity Confirmation — miners, energy, materials, batteries, hydrogen
    "Critical_Minerals_RareEarths": ["Commodity_Confirmation"],
    "Copper_Supercycle":            ["Commodity_Confirmation"],
    "Nuclear_Energy":               ["Commodity_Confirmation"],
    "Solar_Energy":                 ["Commodity_Confirmation"],
    "EV_Batteries":                 ["Commodity_Confirmation"],
    "Hydrogen_Fuel_Cells":          ["Commodity_Confirmation"],

    # Idiosyncratic — single-narrative themes
    "Cybersecurity_FounderLed":     ["Idiosyncratic"],
    "Insurance_InsureTech":         ["Idiosyncratic"],
}

# =============================================================================
# REVERSE LOOKUP: macro group → list of themes in that group
# =============================================================================

MACRO_GROUPS = {}
for theme, groups in THEME_TO_MACRO_GROUPS.items():
    for grp in groups:
        MACRO_GROUPS.setdefault(grp, []).append(theme)


# =============================================================================
# UNIQUE TICKER UNIVERSE (for batch fetching)
# =============================================================================

def get_all_unique_tickers():
    """Returns sorted list of every unique ticker across all themes."""
    s = set()
    for tickers in THEMES.values():
        s.update(tickers)
    return sorted(s)


def get_themes_for_ticker(ticker):
    """Returns list of theme names that contain a given ticker."""
    ticker = ticker.upper()
    return [name for name, tickers in THEMES.items() if ticker in tickers]


def get_macro_groups_for_theme(theme_name):
    """Returns list of macro groups for a given theme."""
    return THEME_TO_MACRO_GROUPS.get(theme_name, [])
