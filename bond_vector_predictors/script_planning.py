        training = np.loadtxt(".txt", usecols=range(1, 79))
        ### Load in testing set which contains information for only 2l3e ###
        testing = np.loadtxt(".txt", usecols=range(1, 79))
        ### Divide training and testing sets into s2 order parameters and features ###
        self.y_train = training[:, 0]  # s2 order parameters for all RNAs except 2l3e
        self.x_train = training[:, 1:]  # features for all RNAs except 2l3e
        self.y_test = testing[:, 0]  # s2 order parameters for 2l3e only
        self.x_test = testing[:, 1:]  # features for 2l3e only
        ### Obtain a list of resids ###
        resid = np.loadtxt("2l3e.test.txt", usecols=range(80))
        for v in resid:
            self.all_resid.append(v)
        ### Obtain a list of measured s2 order parameters ###
        for i in self.y_test:
            self.all_measured_s2.append(i)
        # print self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape
        
        
        
    def plot_RF_s2_resid(self):
        """A method for plotting s2 Order Parameter vs Residue Index"""
        x = self.all_resid[0:35]
        y1 = self.all_RF_predicted_s2
        y2 = self.all_measured_s2
        plt.title('s2 Order Parameters vs. Residue Index')
        plt.plot(x,y1, marker='o', linestyle='-', color='r')
        plt.plot(x,y2, marker='o', linestyle='-', color='b')
        plt.xlabel('Residue Index')
        plt.ylabel('s2 Order Parameters')
        red_patch = mpatches.Patch(color='red', label='Predicted s2 Order Parameters for 2l3e (RF)')
        blue_patch = mpatches.Patch(color='blue', label='Measured s2 Order Parameters for 2l3e')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

    def plot_ET_s2_resid(self):
        """A method for plotting s2 Order Parameter vs Residue Index"""
        x = self.all_resid[0:35]
        y1 = self.all_ET_predicted_s2
        y2 = self.all_measured_s2
        plt.title('s2 Order Parameters vs. Residue Index')
        plt.plot(x,y1, marker='o', linestyle='-', color='r')
        plt.plot(x,y2, marker='o', linestyle='-', color='b')
        plt.xlabel('Residue Index')
        plt.ylabel('s2 Order Parameters')
        red_patch = mpatches.Patch(color='red', label='Predicted s2 Order Parameters for 2l3e (ET)')
        blue_patch = mpatches.Patch(color='blue', label='Measured s2 Order Parameters for 2l3e')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()
        
        def plot_RF_s2_diff_resid(self):
        """Plotting difference in s2 Order Parameters (measured - predicted) vs. Residue Index"""
        x = self.all_resid[0:35]
        y = self.all_RF_s2_diff
        axhline(linewidth=1.5, color='r')
        plt.title('Difference in s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        plt.plot(x, y, marker='None', linestyle='-', color='r')
        blue_line = mlines.Line2D([], [], color='blue', marker='o', markersize=8,
                              label='Measured - Predicted s2 Order Parameters (RF)')
        plt.legend(handles=[blue_line])
        plt.xlabel('Residue Index')
        plt.ylabel('Difference in s2 Order Parameters')
        plt.show()

    def plot_ET_s2_diff_resid(self):
        """Plotting difference in s2 Order Parameters (measured - predicted) vs. Residue Index"""
        x = self.all_resid[0:35]
        y = self.all_ET_s2_diff
        axhline(linewidth=1.5, color='r')
        plt.title('Difference in s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        plt.plot(x, y, marker='None', linestyle='-', color='r')
        blue_line = mlines.Line2D([], [], color='blue', marker='o', markersize=8,
                              label='Measured - Predicted s2 Order Parameters (ET)')
        plt.legend(handles=[blue_line])
        plt.xlabel('Residue Index')
        plt.ylabel('Difference in s2 Order Parameters')
        plt.show()


			
	def make_table_RF(self):
		"""A method for making a table of all measured s2 order parameters, predicted s2 order parameters,
		difference in s2 order parameters, and feature importances for the random forest model"""
		for i, t, j, l in zip(self.all_measured_s2, self.all_RF_predicted_s2, self.all_RF_s2_diff,
								self.RF_importances_list):
			out = "%s %s %s %s" %(i, t, j, l)
			print out
	
	def make_table_ET(self):		
		"""A method for making a table of all measured s2 order parameters, predicted s2 order parameters,
		difference in s2 order parameters, and feature importances for the extra trees model"""
		for i, t, j, l in zip(self.all_measured_s2, self.all_ET_predicted_s2, self.all_ET_s2_diff,
								self.ET_importances_list):
			out = "%s %s %s %s" %(i, t, j, l)
			print out
			
			
			### Obtain a list of resids ###
		#resid = np.loadtxt("10.0A_all_C1.txt", usecols= range(79,80))
		#for v in resid:
			#self.all_resid.append(v)
	def plot_s2_resid(self):
		"""A method for plotting s2 Order Parameter vs. Residue Index"""
		x = self.all_resid
		y1 = self.all_LR_predicted_s2
		y2 = self.all_RF_predicted_s2
		y3 = self.all_ET_predicted_s2
		y4 = self.all_measured_s2
		plt.title('s2 Order Parameters vs. Residue Index')
		plt.plot(x,y1, marker='o', linestyle='-', color='b')
		plt.plot(x,y2, marker='o', linestyle='-', color='k')
		plt.plot(x,y3, marker='o', linestyle='-', color='g')
		plt.plot(x,y4, marker='o', linestyle='-', color='r')
		plt.xlabel('Residue Index')
		plt.ylabel('s2 Order Parameters')
		blue_patch = mpatches.Patch(color='blue', label='Predicted s2 Order Parameters for 2l3e (LR)')
		black_patch = mpatches.Patch(color='black', label='Predicted s2 Order Parameters for 2l3e (RF)')
		green_patch = mpatches.Patch(color='green', label='Predicted s2 Order Parameters for 2l3e (ET)')
		red_patch = mpatches.Patch(color='red', label='Measured s2 Order Parameters for 2l3e')
		plt.legend(handles=[red_patch, blue_patch, black_patch, green_patch])
		plt.show()
		
	def plot_s2_diff_resid(self):
		"""Plotting difference in s2 Order Parameters (measured - predicted) vs. Residue Index"""
		x = self.all_resid
		y1 = self.all_LR_s2_diff
		y2 = self.all_RF_s2_diff
		y3 = self.all_ET_s2_diff
		axhline(linewidth =1.5, color='r')
		plt.title('Difference in s2 Order Parameters vs. Residue Index')
		plt.plot(x,y1, marker='o', linestyle='-', color='b')
		plt.plot(x,y2, marker='o', linestyle='-', color='k')
		plt.plot(x,y3, marker='o', linestyle='-', color='g')
		blue_line = mlines.Line2D([],[], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (LR)')
		black_line = mlines.Line2D([],[], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (RF)')
		green_line = mlines.Line2D([],[], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (ET)')
		plt.legend(handles=[blue_line, black_line, green_line])
		plt.xlabel('Residue Index')
		plt.ylabel('Difference in s2 Order Parameters')
		plt.show()

	self.x_train = self.x_train[:, 1:77] # training features, resid NOT included
		self.x_test = self.x_test[:, 1:77] 
			### Obtain a list of testing set resids ###
			resid = self.x_test[:, 77]
		for v in resid:
			self.resid_list.append(v)
			
			
	self.resid_list = []
	
		
	def make_table_ET(self):		
		"""A method for making a table of all resids, measured s2 order parameters, predicted s2 order parameters,
		difference in s2 order parameters, and regression coefficients for the linear regression model"""
		for v, i, t, j, l in zip(self.resid_list, self.all_measured_s2, self.all_ET_predicted_s2, self.all_ET_s2_diff,
								self.ET_importances_list):
			out = "%s %s %s %s %s" %(v, i, t, j, l)
			print out
			
			
				### Make Plots ###
	#predictor.plot_s2_resid()
	#predictor.plot_s2_diff_resid()