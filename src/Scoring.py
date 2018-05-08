import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import json

