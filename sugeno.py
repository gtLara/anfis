import numpy as np
from matplotlib import pyplot as plt

class Sugeno():

    def __init__(self, antecedents, consequents):
        #fiquei incomodado tmb quando percebi
        assert(len(consequents) == len(antecedents))
        self.consequents = consequents
        self.antecedents = antecedents

    def infer(self, x):
        weights = [ant.infer(x) for ant in self.antecedents]
        consequent_values = [con.infer(x) for con in self.consequents]

        output = 0
        for (w, cm) in zip(weights, consequent_values):
            output += w*cm

        normal_output = output/sum(weights)

        return normal_output

    def inferbycurve(self):
        weights = self.antecedents
        consequent_values = self.consequents

        output = 0
        for (w, cm) in zip(weights, consequent_values):
            output += w*cm

        normal_output = output/sum(weights)

        return normal_output