'''
Created on Mar 10, 2014

@author: lewhoo
'''


import numpy as np

## Class for working with the Experiment TTree			
class ExpTree():

	## Contructor
	def __init__(self, tree=None, tree_file=None, old_tree_type=False):
		self.old_tree_type=old_tree_type
		# If no Trees have been provided, assume EUSO-TA config
		if tree==None:
			self.ccbCount = 1
			self.pdmCount = 1
			self.pmtCountX = 6
			self.pmtCountY = 6
			self.pixelCountX = 8
			self.pixelCountY = 8
			
			# Initialize and fill the focal surface shape
			self.FocalsurfaceMaxPdmX = 1
			self.FocalsurfaceMaxPdmY = 1
			self.Focalsurface = np.array([1], dtype=np.bool).reshape(self.FocalsurfaceMaxPdmX, self.FocalsurfaceMaxPdmY)
	
			# Derived values for simpler use
			
			# The counts of pixels in PDMs
			self.pdmPixelCountX = self.pixelCountX*self.pmtCountX
			self.pdmPixelCountY = self.pixelCountY*self.pmtCountY
			
			# The maximal possible coordinates of histogram pixel (and thus focal surface pixel)
			self.MaxHistoPixelX = self.pixelCountX*self.FocalsurfaceMaxPdmX*self.pmtCountX
			self.MaxHistoPixelY = self.pixelCountY*self.FocalsurfaceMaxPdmY*self.pmtCountY
			
			self.map_phys2el = np.zeros((self.pdmPixelCountX, self.pdmPixelCountY), dtype='I')
			self.map_el2phys_x = np.zeros(self.pdmPixelCountX*self.pdmPixelCountY)
			self.map_el2phys_y = np.zeros(self.pdmPixelCountX*self.pdmPixelCountY)
			self.pmt_map_phys2el = np.zeros((self.pmtCountX, self.pmtCountY), dtype='I')
			self.pmt_map_el2phys_x = np.zeros(self.pmtCountX*self.pmtCountY)
			self.pmt_map_el2phys_y = np.zeros(self.pmtCountX*self.pmtCountY)
			self.gain_map_phys2el = np.zeros((self.pdmPixelCountX, self.pdmPixelCountY), dtype='I')
			self.gain_map_el2phys_x = np.zeros(self.pdmPixelCountX*self.pdmPixelCountY)
			self.gain_map_el2phys_y = np.zeros(self.pdmPixelCountX*self.pdmPixelCountY)


			
		# Standard case, read info from the Tree
		else:
			self.tree = tree
			self.tree_file = tree_file
			# Initialize values from the tree
			self.tree.GetEntry()
			self.ccbCount = self.tree.ccb_count
			self.pdmCount = self.tree.pdm_count
			self.pmtCountX = self.tree.pmt_count_x
			self.pmtCountY = self.tree.pmt_count_y
			self.pixelCountX = self.tree.pixel_count_x
			self.pixelCountY = self.tree.pixel_count_y
			self.frameCount = self.tree.frame_count
			self.experimentName = self.tree.experiment_name
			
			self.run_mode = self.tree.run_mode
			
			# Initialize and fill the focal surface shape
			self.FocalsurfaceMaxPdmX = self.tree.focal_surface_max_pdm_x
			self.FocalsurfaceMaxPdmY = self.tree.focal_surface_max_pdm_y
			self.Focalsurface = np.array(self.tree.focal_surface, dtype=np.bool).reshape(self.FocalsurfaceMaxPdmX, self.FocalsurfaceMaxPdmY)
	
			# Derived values for simpler use
			
			# The counts of pixels in PDMs
			self.pdmPixelCountX = self.pixelCountX*self.pmtCountX
			self.pdmPixelCountY = self.pixelCountY*self.pmtCountY
			
			# The maximal possible coordinates of histogram pixel (and thus focal surface pixel)
			self.MaxHistoPixelX = self.pixelCountX*self.FocalsurfaceMaxPdmX*self.pmtCountX
			self.MaxHistoPixelY = self.pixelCountY*self.FocalsurfaceMaxPdmY*self.pmtCountY

			self.map_phys2el = np.zeros((self.pdmPixelCountX, self.pdmPixelCountY), dtype='I')
			self.pmt_map_phys2el = np.zeros((self.pmtCountX, self.pmtCountY), dtype='I')
			self.gain_map_phys2el = np.zeros((self.pdmPixelCountX, self.pdmPixelCountY), dtype='I')
			if self.tree.GetBranch("map_el2phys_x")!=None:
				self.map_el2phys_x = self.tree.map_el2phys_x
				self.map_el2phys_y = self.tree.map_el2phys_y
				self.tree.SetBranchAddress("map_phys2el", self.map_phys2el)
				self.pmt_map_el2phys_x = self.tree.pmt_map_el2phys_x
				self.pmt_map_el2phys_y = self.tree.pmt_map_el2phys_y
				self.tree.SetBranchAddress("pmt_map_phys2el", self.pmt_map_phys2el)
				# Not there for older trees
				if hasattr(self.tree, "gain_map_el2phys_x"):
					self.gain_map_el2phys_x = self.tree.gain_map_el2phys_x
					self.gain_map_el2phys_y = self.tree.gain_map_el2phys_y
					self.tree.SetBranchAddress("gain_map_phys2el", self.gain_map_phys2el)
			else:
				self.map_phys2el = np.zeros((self.pdmPixelCountX, self.pdmPixelCountY), dtype='I')
				self.map_el2phys_x = np.zeros(self.pdmPixelCountX*self.pdmPixelCountY)
				self.map_el2phys_y = np.zeros(self.pdmPixelCountX*self.pdmPixelCountY)
				self.pmt_map_phys2el = np.zeros((self.pmtCountX, self.pmtCountY), dtype='I')
				self.pmt_map_el2phys_x = np.zeros(self.pmtCountX*self.pmtCountY)
				self.pmt_map_el2phys_y = np.zeros(self.pmtCountX*self.pmtCountY)
				self.gain_map_phys2el = np.zeros((self.pdmPixelCountX, self.pdmPixelCountY), dtype='I')
				self.gain_map_el2phys_x = np.zeros(self.pdmPixelCountX*self.pdmPixelCountY)
				self.gain_map_el2phys_y = np.zeros(self.pdmPixelCountX*self.pdmPixelCountY)

		self.pdmPixelCount = self.pdmPixelCountX*self.pdmPixelCountY
		self.tree.GetEntry(0)