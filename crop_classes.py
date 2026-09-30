"""LUCAS LC1 land-cover codes mapped to the 17 crop classes of the Copernicus HRL Cropland legend.

Key: LUCAS LC1 code. Value: (CLMS class code, class name). LC1 codes not listed are not crops of
the legend and are discarded.
"""
CLMS_MAP = {
    'B11': ('1110', 'Wheat'), 'B12': ('1110', 'Wheat'), 'B13': ('1120', 'Barley'),
    'B16': ('1130', 'Maize'), 'B17': ('1140', 'Rice'), 'B14': ('1150', 'Other cereals'),
    'B15': ('1150', 'Other cereals'), 'B18': ('1150', 'Other cereals'), 'B19': ('1150', 'Other cereals'),
    'B42': ('1210', 'Fresh vegetables'), 'B43': ('1210', 'Fresh vegetables'), 'B41': ('1220', 'Dry pulses'),
    'B21': ('1310', 'Potatoes'), 'B22': ('1320', 'Sugar Beet'), 'B31': ('1410', 'Sunflower'),
    'B33': ('1420', 'Soybeans'), 'B32': ('1430', 'Rapeseed'), 'B34': ('1440', 'Flax, cotton and hemp'),
    'B82': ('2100', 'Grapes'), 'B81': ('2200', 'Olives'), 'B71': ('2310', 'Fruits'),
    'B72': ('2310', 'Fruits'), 'B73': ('2310', 'Fruits'), 'B75': ('2310', 'Fruits'),
    'B76': ('2310', 'Fruits'), 'B77': ('2310', 'Fruits'), 'B74': ('2320', 'Nuts'),
}
