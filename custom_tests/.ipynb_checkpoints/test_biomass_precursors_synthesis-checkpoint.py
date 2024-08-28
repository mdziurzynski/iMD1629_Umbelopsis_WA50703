import unittest
import pandas
import cobra
import os

from cobra.io import read_sbml_model, write_sbml_model, validate_sbml_model
from cobra import Model, Reaction, Metabolite
from cobra.flux_analysis.reaction import assess_component
from cobra.util.array import create_stoichiometric_matrix


class TestBiomassPrecursorsSynthesisOnMinimalMediumWithGlucose(unittest.TestCase):
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
        
        both_bounds_to_open = ["EX_956_e", "EX_1407_e","EX_1782_e","EX_1503_e","EX_1665_e","EX_1544_e","EX_118_e","EX_44_e","EX_1653_e"]
        for rxn_id in both_bounds_to_open:
            rxn = self.model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = -1000
            rxn.upper_bound = 1000
        

    def test_protein_synthesis(self):
        model = self.model.copy()
        model.objective = "Protein_synthesis"
        model.add_boundary(model.metabolites.get_by_id("Protein_c"), type="demand")
        solution = model.optimize()
        self.assertTrue(solution.objective_value > 0.1)
        
        # test exchanges - check if the mets are in fact inported
        # right now I test for exchanges that are used in the newest version of the model
        ex_fluxes = solution.fluxes.loc[['EX_956_e', "EX_1503_e"]]
        
        self.assertTrue( (ex_fluxes < -1).all() )
        
    def test_DNA_synthesis(self):
        model = self.model.copy()
        model.objective = "DNA_synthesis"
        model.add_boundary(model.metabolites.get_by_id("DNA_c"), type="demand")
        solution = model.optimize()
        self.assertTrue(solution.objective_value > 0.1)
        
        # test exchanges - check if the mets are in fact imported
        # right now I test for exchanges that are used in the newest version of the model
        ex_fluxes = solution.fluxes.loc[['EX_956_e', "EX_1503_e"]]
        
        self.assertTrue( (ex_fluxes < -1).all() )
        
    def test_RNA_synthesis(self):
        model = self.model.copy()
        model.objective = "RNA_synthesis"
        model.add_boundary(model.metabolites.get_by_id("RNA_c"), type="demand")
        solution = model.optimize()
        self.assertTrue(solution.objective_value > 0.1)
        
        ex_fluxes = solution.fluxes.loc[['EX_956_e', "EX_1503_e"]]
        
        
    def test_carbohydrates_synthesis(self):
        model = self.model.copy()
        model.objective = "Carbohydrates_synthesis"
        model.add_boundary(model.metabolites.get_by_id("Carbohydrates_c"), type="demand")
        solution = model.optimize()
        self.assertTrue(solution.objective_value > 0.1)
        
        ex_fluxes = solution.fluxes.loc[['EX_956_e']]
        
        self.assertTrue( (ex_fluxes < -1).all() )
        
    def test_free_fatty_acids_synthesis(self):
        model = self.model.copy()
        model.objective = "free_fatty_acids_formation"
        model.add_boundary(model.metabolites.get_by_id("generic_fatty_acid_c"), type="demand")
        solution = model.optimize()
        self.assertTrue(solution.objective_value > 0.1)
        
        ex_fluxes = solution.fluxes.loc[['EX_956_e']]
        
        self.assertTrue( (ex_fluxes < -1).all() )
        
        
    def test_neutral_lipids_synthesis(self):
        model = self.model.copy()
        model.objective = "Neutral_lipids_synthesis"
        model.add_boundary(model.metabolites.get_by_id("Neutral_lipids_c"), type="demand")
        solution = model.optimize()
        self.assertTrue(solution.objective_value > 0.1)
        
        ex_fluxes = solution.fluxes.loc[['EX_956_e']]
        
        self.assertTrue( (ex_fluxes < -1).all() )
        
    def test_phospholipids_synthesis(self):
        model = self.model.copy()
        model.objective = "Phospholipids_synthesis"
        model.add_boundary(model.metabolites.get_by_id("Phospholipids_c"), type="demand")
        solution = model.optimize()
        self.assertTrue(solution.objective_value > 0.1)
        
        ex_fluxes = solution.fluxes.loc[['EX_956_e']]
        
        self.assertTrue( (ex_fluxes < -1).all() )
        
    def test_biomass_synthesis(self):
        model = self.model.copy()
        model.objective = "Biomass_reaction_1"
        solution = model.optimize()
        self.assertTrue(solution.objective_value > 0.04)
        
        # test exchanges - check if the mets are in fact inported
        # right now I test for exchanges that are used in the newest version of the model
        ex_fluxes = solution.fluxes.loc[["EX_1503_e", "EX_1544_e", "EX_1653_e", "EX_44_e", "EX_956_e"]]
        
        self.assertTrue( (ex_fluxes < -0.0001).all() )

if __name__ == '__main__':
    unittest.main()