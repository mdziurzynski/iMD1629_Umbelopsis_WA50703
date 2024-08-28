import unittest
import pandas as pd
import cobra
import os
from tqdm.notebook import tqdm

from cobra.io import read_sbml_model, write_sbml_model, validate_sbml_model
from cobra import Model, Reaction, Metabolite
from cobra.flux_analysis.reaction import assess_component
from cobra.util.array import create_stoichiometric_matrix

class TestBiologExperimentalDataGrowth(unittest.TestCase):
    
    def get_newest_model_version():        
        
        path_to_model = "../iMD1629.xml"
        print(f"Testing on model: {path_to_model}")
        
        return path_to_model
    
    
    @classmethod
    def setUpClass(self):
        self.path_to_model = self.get_newest_model_version()
        cobra_config = cobra.Configuration()
        cobra_config = "gurobi"
        self.model = read_sbml_model(self.path_to_model)
        
        # close all exchanges
        for rxn in self.model.boundary:
            rxn.lower_bound = 0
            rxn.upper_bound = 0
        
        # open exchanges for minimal medium
        upper_bound_to_open = ["DM_Biomass_c","DM_1059_m", "DM_148_c", "DM_1111_e"]
        for rxn_id in upper_bound_to_open:
            rxn = self.model.reactions.get_by_id(rxn_id)
            rxn.upper_bound = 1000
        
        both_bounds_to_open = ["EX_1407_e","EX_1782_e","EX_1503_e","EX_1665_e","EX_1544_e","EX_118_e","EX_44_e","EX_1653_e"]
        for rxn_id in both_bounds_to_open:
            rxn = self.model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = -1000
            rxn.upper_bound = 1000
            
    
    def test_no_growth_without_carbon_source(self):
        model = self.model.copy()
        model.objective = "Biomass_reaction_1"
        solution = model.optimize()
        self.assertTrue(solution.status ==  "infeasible")

    def test_biolog_carbon_source_1361_e(self):
        "Metabolite name: N-acetyl-D-glucosamine[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1361_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_960_e(self):
        "Metabolite name: D-ribulose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("960_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_968_c(self):
        "Metabolite name: D-arabinofuranose[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("968_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 0)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1184_e(self):
        "Metabolite name: L-arabinose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1184_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1185_e(self):
        "Metabolite name: L-arabinitol[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1185_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_132_e(self):
        "Metabolite name: beta-D-cellobiose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("132_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_994_c(self):
        "Metabolite name: erythritol[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("994_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 0)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_95_e(self):
        "Metabolite name: beta-D-fructofuranose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("95_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1029_e(self):
        "Metabolite name: D-galactose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1029_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_954_e(self):
        "Metabolite name: D-galactopyranuronate[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("954_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1047_c(self):
        "Metabolite name: D-gluconate[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1047_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_glucosamine_e(self):
        "Metabolite name: D-glucosamine[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("glucosamine_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_956_e(self):
        "Metabolite name: D-glucose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("956_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1039_e(self):
        "Metabolite name: alpha-D-glucopyranose 1-phosphate[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1039_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_955_e(self):
        "Metabolite name: D-glucopyranuronate[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("955_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 0)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1056_e(self):
        "Metabolite name: glycerol[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1056_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1269_c(self):
        "Metabolite name: myo-inositol[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1269_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 0)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1242_e(self):
        "Metabolite name: maltose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1242_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1244_e(self):
        "Metabolite name: maltotriose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1244_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_971_e(self):
        "Metabolite name: D-mannopyranose[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("971_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1252_c(self):
        "Metabolite name: melibiose[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1252_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_75_c(self):
        "Metabolite name: an alpha-D-galactoside[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("75_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 0)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_106_c(self):
        "Metabolite name: a beta-D-galactoside[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("106_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 0)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_231_e(self):
        "Metabolite name: raffinose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("231_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1654_e(self):
        "Metabolite name: D-sorbitol[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1654_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_816_c(self):
        "Metabolite name: L-sorbopyranose[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("816_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_445_c(self):
        "Metabolite name: stachyose[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("445_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1724_e(self):
        "Metabolite name: alpha,alpha-trehalose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1724_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1789_c(self):
        "Metabolite name: xylitol[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1789_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_967_e(self):
        "Metabolite name: D-xylose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("967_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1936_c(self):
        "Metabolite name: 4-aminobutanoate[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1936_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 0)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1020_c(self):
        "Metabolite name: fumarate[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1020_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_490_c(self):
        "Metabolite name: (S)-3-hydroxybutanoate[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("490_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1939_e(self):
        "Metabolite name: 4-hydroxybutanoate[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1939_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1869_c(self):
        "Metabolite name: 2-oxoglutarate[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1869_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1203_c(self):
        "Metabolite name: (S)-lactate[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1203_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1245_c(self):
        "Metabolite name: (S)-malate[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1245_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_798_c(self):
        "Metabolite name: succinamate[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("798_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1663_c(self):
        "Metabolite name: succinate[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1663_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_6_c(self):
        "Metabolite name: N-acetyl-L-glutamate[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("6_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 0)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1183_e(self):
        "Metabolite name: L-alanine[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1183_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_54_e(self):
        "Metabolite name: L-asparagine[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("54_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1188_c(self):
        "Metabolite name: L-aspartate[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1188_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1046_e(self):
        "Metabolite name: L-glutamate[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1046_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1205_e(self):
        "Metabolite name: L-ornithine[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1205_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1450_c(self):
        "Metabolite name: L-phenylalanine[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1450_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 0)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1507_c(self):
        "Metabolite name: L-proline[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1507_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1971_c(self):
        "Metabolite name: 5-oxo-L-proline[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1971_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1651_c(self):
        "Metabolite name: L-serine[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1651_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1716_c(self):
        "Metabolite name: L-threonine[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1716_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_999_c(self):
        "Metabolite name: ethanolamine[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("999_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 0)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1511_e(self):
        "Metabolite name: putrescine[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1511_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_18_e(self):
        "Metabolite name: adenosine[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("18_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_1756_c(self):
        "Metabolite name: uridine[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("1756_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 0)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_45_c(self):
        "Metabolite name: AMP[c]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("45_c")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_arbutrin_e(self):
        "Metabolite name: arbutrin[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("arbutrin_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_gentiobiose_e(self):
        "Metabolite name: gentiobiose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("gentiobiose_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_glycogen_e(self):
        "Metabolite name: glycogen[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("glycogen_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_diketodgluconate_e(self):
        "Metabolite name: 2-keto-D-gluconate[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("diketodgluconate_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_lactose_e(self):
        "Metabolite name: Lactose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("lactose_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_lactulose_e(self):
        "Metabolite name: Lactulose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("lactulose_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_maltitol_e(self):
        "Metabolite name: maltitol[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("maltitol_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_mannitol_e(self):
        "Metabolite name: mannitol[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("mannitol_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_melezitose_e(self):
        "Metabolite name: melezitose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("melezitose_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_palatinose_e(self):
        "Metabolite name: palatinose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("palatinose_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_961_e(self):
        "Metabolite name: D-ribofuranose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("961_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_turanose_e(self):
        "Metabolite name: turanose[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("turanose_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_alaninamide_e(self):
        "Metabolite name: alaninamide[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("alaninamide_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_ala_gly_e(self):
        "Metabolite name: L-alanyl-glycine[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("ala_gly_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()

    def test_biolog_carbon_source_glycyl_l_glutamate_e(self):
        "Metabolite name: glycyl-L-glutamate[e]"
        model = self.model
        model.objective = "Biomass_reaction_1"
        met = model.metabolites.get_by_id("glycyl_l_glutamate_e")
        sk_met = model.add_boundary(met, type="sink")
        sol = model.optimize()
        
        if sol.status == "infeasible":
            result = 0
        elif sol.status == "optimal" and sol.objective_value > 0.0000001:
            result = 1
        self.assertTrue(result == 1)
        sk_met.remove_from_model()