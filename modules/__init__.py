"""
Tanker Synthesis Model — v0.1
INDOPACOM Distributed Logistics Product Tanker Design Family
"""
from .design_point import DesignPoint, HullParams, MissionParams, PropulsionParams, make_design_family
from .resistance_holtrop import HoltropMennen, PropulsiveCoefficients, compute_resistance
from .weights_watson import WatsonWeights, compute_weights
from .stability_prelim import PrelimStability, compute_stability
from .cost_parametric import ParametricCost, compute_cost
