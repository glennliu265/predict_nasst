#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:34:22 2023

@author: gliu
"""



f1 = "/Users/gliu/ParamTesting/nlayers6_nunits256_dropoutTrue/Models/SSH_lead25_classify.pt"
f2 = "/Users/gliu/ParamTesting/nlayers4_nunits64_dropoutTrue/Models/SSH_lead25_classify.pt"


m265=torch.load(f1)
m64 = torch.load(f2)