'use strict';
window.defenderModelNames = Object.freeze({
  "A": "A · Whole-blood neural regression",
  "B": "B · WBC-pretrained neural regression",
  "C": "C · Multitask neural network for exposure and dose",
  "D": "D · Weighted neural regression with WBC and whole blood",
  "elastic_net": "Elastic Net regression",
  "pls": "Partial least squares regression",
  "gradient_boosting": "Gradient boosting (whole blood only)",
  "gradient_boosting_wb_wbc": "Gradient boosting (WBC + whole blood data)",
  "median_null": "Training-median baseline"
});
