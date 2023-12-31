pathNames = [
             '000000',
             '000001',
             '500001','050001','005001','000501','000051',
             '550001','505001','500501','500051',
             '055001','050501','050051',
             '005501','005051',
             '000551',
             '555001','550501','550051',
             '055501','055051',
             '005551',
             '555501','555051','550551','505551','055551',
             '555551',
             'X00001','0X0001','00X001','000X01','0000X1',
             'XX0001','X0X001','X00X01','X000X1','0XX001','0X0X01','0X00X1','00XX01','00X0X1','000XX1',
             'XXX001','XX0X01','XX00X1',
             '0XXX01','0XX0X1','00XXX1',
             'XXXX01','XXX0X1','XX0XX1','X0XXX1','0XXXX1',
             'XXXXX1',
             'X55551','5X5551','55X551','555X51','5555X1','XX5551','X5X551','X55X51','X555X1',
             ]


labelMap = {
            '000000':[0,    0,    0,    0,    0,    0],
            '000001':[0,    0,    0,    0,    0,    1],
            '500001':[0.5,  0,    0,    0,    0,    1],
            '050001':[0,    0.5,  0,    0,    0,    1],
            '005001':[0,    0,    0.5,  0,    0,    1],
            '000501':[0,    0,    0,    0.5,  0,    1],
            '000051':[0,    0,    0,    0,    0.5,  1],
            '550001':[0.5,  0.5,  0,    0,    0,    1],
            '505001':[0.5,  0,    0.5,  0,    0,    1],
            '500501':[0.5,  0,    0,    0.5,  0,    1],
            '500051':[0.5,  0,    0,    0,    0.5,  1],
            '055001':[0,    0.5,  0.5,  0,    0,    1],
            '050501':[0,    0.5,  0,    0.5,  0,    1],
            '050051':[0,    0.5,  0,    0,    0.5,  1],
            '005501':[0,    0,    0.5,  0.5,  0,    1],
            '005051':[0,    0,    0.5,  0,    0.5,  1],
            '000551':[0,    0,    0,    0.5,  0.5,  1],
            '555001':[0.5,  0.5,  0.5,  0,    0,    1],
            '550501':[0.5,  0.5,  0,    0.5,  0,    1],
            '550051':[0.5,  0.5,  0,    0,    0.5,  1],
            '055501':[0,    0.5,  0.5,  0.5,  0,    1],
            '055051':[0,    0.5,  0.5,  0,    0.5,  1],
            '005551':[0,    0,    0.5,  0.5,  0.5,  1],
            '555501':[0.5,  0.5,  0.5,  0.5,  0,    1],
            '555051':[0.5,  0.5,  0.5,  0,    0.5,  1],
            '550551':[0.5,  0.5,  0,    0.5,  0.5,  1],
            '505551':[0.5,  0,    0.5,  0.5,  0.5,  1],
            '055551':[0,    0.5,  0.5,  0.5,  0.5,  1],
            '555551':[0.5,  0.5,  0.5,  0.5,  0.5,  1],
            'X00001':[1,    0,    0,    0,    0,    1],
            '0X0001':[0,    1,    0,    0,    0,    1],
            '00X001':[0,    0,    1,    0,    0,    1],
            '000X01':[0,    0,    0,    1,    0,    1],
            '0000X1':[0,    0,    0,    0,    1,    1],
            'XX0001':[1,    1,    0,    0,    0,    1],
            'X0X001':[1,    0,    1,    0,    0,    1],
            'X00X01':[1,    0,    0,    1,    0,    1],
            'X000X1':[1,    0,    0,    0,    1,    1],
            '0XX001':[0,    1,    1,    0,    0,    1],
            '0X0X01':[0,    1,    0,    1,    0,    1],
            '0X00X1':[0,    1,    0,    0,    1,    1],
            '00XX01':[0,    0,    1,    1,    0,    1],
            '00X0X1':[0,    0,    1,    0,    1,    1],
            '000XX1':[0,    0,    0,    1,    1,    1],
            'XXX001':[1,    1,    1,    0,    0,    1],
            'XX0X01':[1,    1,    0,    1,    0,    1],
            'XX00X1':[1,    1,    0,    0,    1,    1],
            '0XXX01':[0,    1,    1,    1,    0,    1],
            '0XX0X1':[0,    1,    1,    0,    1,    1],
            '0X0XX1':[0,    1,    0,    1,    1,    1],
            '00XXX1':[0,    0,    1,    1,    1,    1],
            'XXXX01':[1,    1,    1,    1,    0,    1],
            'XXX0X1':[1,    1,    1,    0,    1,    1],
            'XX0XX1':[1,    1,    0,    1,    1,    1],
            'X0XXX1':[1,    0,    1,    1,    1,    1],
            '0XXXX1':[0,    1,    1,    1,    1,    1],
            'XXXXX1':[1,    1,    1,    1,    1,    1],
            'X55551':[1,    0.5,  0.5,  0.5,  0.5,  1],
            '5X5551':[0.5,  1,    0.5,  0.5,  0.5,  1],
            '55X551':[0.5,  0.5,  1,    0.5,  0.5,  1],
            '555X51':[0.5,  0.5,  0.5,  1,    0.5,  1],
            '5555X1':[0.5,  0.5,  0.5,  0.5,  1,    1],
            'XX5551':[1,    1,    0.5,  0.5,  0.5,  1],
            'X5X551':[1,    0.5,  1,    0.5,  0.5,  1],
            'X55X51':[1,    0.5,  0.5,  1,    0.5,  1],
            'X555X1':[1,    0.5,  0.5,  0.5,  1,    1],
            }

