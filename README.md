
QNPy Documentation
==================

Introduction
============
In exploring the diverse features of quasar light curves, a significant challenge arises from recurring gaps in observations, which pose a primary limitation. This obstacle, compounded by the inherent irregularities in data collection cadences, presents a formidable barrier. This complexity will be particularly pronounced when dealing with data that is going to come from the Legacy Survey of Space and Time (LSST), featuring seasonal gaps. Existing strategies, while effective, entail substantial computational costs. To address the complex nature of quasar light curve modeling, our package QNPy has been developed to efficiently model quasar light curves using Conditional Neural Processes.

Conditional Neural Processes
----------------------------

Conditional Neural Processes (CNPs) are a type of neural network architecture designed for flexible and probabilistic function learning. They are particularly well-suited for tasks involving conditional predictions and have applications in areas like regression, classification, and generative modeling.

The core idea behind CNPs is to learn a distribution over functions conditioned on input-output pairs. They are capable of making predictions not only for specific inputs seen during training but also for new inputs that were not present in the training data.

The CNP is a model designed for analyzing continuous-time light curves, characterized by time instances (x) and corresponding fluxes (or magnitudes) (y). In the CNP framework, we consider a scenario where we have target inputs representing time instances with unknown magnitudes. In the training process, we leverage a set of context points derived from observations, consisting of time instances (x) and observed magnitudes (y). Each pair in the context set is locally encoded using a multilayer perceptron (MLP). The resulting local encodings (Rc) are then aggregated through mean pooling to form a global representation (R). The global representation (R) is a condensed feature representation of the entire context set. This representation, along with the target input (xt), is fed into a decoder MLP. The decoder produces the mean and variance of the predictive distribution for the target output.

.. image:: _static/CNP.png
   :alt: descriptive text for image
   :align: left

Key features of Conditional Neural Processes include:

1. **Conditional Predictions**: CNPs can provide predictions for a target output given a set of input-output pairs, allowing for context-aware predictions.

2. **Flexibility**: CNPs are versatile and can adapt to various types of data and tasks. They are not limited to a specific functional form, making them suitable for a wide range of applications.

3. **Probabilistic Outputs**: CNPs provide uncertainty estimates in the form of predictive distributions. This makes them valuable in situations where uncertainty quantification is crucial.

4. **Scalability**: CNPs can handle different input and output dimensions, making them scalable to various data types and problem complexities.

In summary, Conditional Neural Processes are a powerful framework for conditional function learning that offers flexibility, probabilistic predictions, and scalability across different tasks. They have shown effectiveness in tasks such as few-shot learning, meta-learning, and regression with uncertainty estimation, making them a great tool for modeling the light curves of quasars.

Self Organizing Maps 
--------------------

Conditional Neural Processes excel at learning complex patterns in data with recurring gaps. However, application to larger datasets requires novel methods to prioritize efficiency and effectively capture subtle trends in the data. Self Organizing Maps (SOMs) provide both these advantages. SOMs provide an unsupervised clustering algorithm that can be trained quickly and include new data points without the need to train over every data point again. Thus, we present QNPy as an ensemble model of SOMs and CNPs.

SOMs comprise a network of nodes mapped onto a (usually) two-dimensional grid. Each node has an input weight associated with it. As the SOM trains on the input data, each input point is assigned a Best Matching Unit (BMU) where the node is at the minimum Euclidean distance from the input. Then, the BMU is updated to match the input data point (the amount that the node moves is dependent on the learning rate). Furthermore, each node can affect neighboring nodes via a neighborhood function (usually Gaussian). 

Once the training is ended, each input data point is assigned to a cluster depending on the final BMU. Thus at the end, each node provides a cluster. These can be the final cluster or the distance matrix (a matrix containing the distance of each node with each of the other nodes) of the SOM can be used to group different nodes into more hierarchical clusters. This is done by calculating gradients between the nodes until the lowest node is reached. (For more info, refer to [Hamel and Brown](https://homepage.cs.uri.edu/faculty/hamel/pubs/improved-umat-dmin11.pdf)).

In QNPy, we treat each light curve as a data point and the magnitudes are the features. Thus, the SOM can effectively parse topological differences in the light curves. These differences allow the CNP to train on similar light curves and effectively capture subtle differences in the modeled light curves. In addition, the clusters now allow for CNPs to be trained in parallel on smaller batches of data, which allows for a massive speed-up in the training time of the QNPy package.

The SOM is based on the [minisom package](https://github.com/JustGlowing/minisom) which uses NumPy packages to handle the input data. Thus, every input data point must have the same length. We handle this similarly with the CNP by padding all the light curves to the same length. We also scale the light curves to treat different magnitude ranges differently.

Thus, SOMs provide a useful companion to the CNPs to form an ensemble model with improved speed and accuracy in modeling large amounts of light curve data.

Installation
============

To install QNPy, use the following command:

.. code-block:: bash

    pip install QNPy

Requirements
------------

This package contains a `requirements.txt` file with all the dependencies that need to be satisfied before you can use it as a standalone package. To install all requirements at once, navigate to the directory where you downloaded your package (where the `requirements.txt` file is) and run:

.. code-block:: bash

    pip install -r requirements.txt

You are now ready to use the QNPy package.

Special note: If you have python >3.9 on your local machine you will encounter some requirements conflicts with torch and numpy versions. In this case, we recomend creating a virtual enviroment using conda:

.. code-block:: bash

    conda create -n myenv python=3.9

then you have to activate the virtuel enviroment:

.. code-block:: bash

    conda activate "The name of your virtuel enviroment"

After virtual enviroment is activated you can install QNPy and the requirements.txt file in your newly created enviroment.

.. code-block:: virtual enviroment

    pip install QNPy

.. code-block:: virtual enviroment

    pip install -r requirements.txt

Examples
========

Check out the `Tutorial` folder [here](https://github.com/kittytheastronaut/QNPy-0.0.2) for notebooks that guide you through the process of using this package. There will be teo tutorial folders. The "QNPy without clustering: Tutorial" folder includes examples for using each of the modules separately. Additionally, you'll find an example of how your light curves should look in the `Light_curves` folder. The "QNPy with clustering: Tutorial" folder includes examples for using the Clustering_with_SOM module for single band and multiband clustering. You will also find example of how your light curves should look like in the folders `Light_curves` and `Light_curves_Multiband`.

Folder Structure
================

The QNPy automatically creates folders for saving plots, data and saves your trained SOM and CNP. The only requirement for the file structure in SOM module is to save light curves before the module and choose directories to save plots and models during the module's runtime. The files can be saved under any folder as desired and the file name can be given as an input into the loading function.

In the case of multi-band light curves only, the light curves should be saved under a directory (can be named anything) with the filters saved as subfolders. Then, each light curve should be saved as a CSV file with the id as the file name. For example, if you have a light curve in the g filter with ID 10422 and you want to save it in a folder known as `Light_Curves`, it should be saved under the directory `Light_Curves/g/10422.csv`. This is the standard recommendation for multi-band data. Then, once the clusters are created, it is easy to either point QNPy to one of the filters of a cluster or to manually flatten the file and provide all the light curves to QNPy. However, QNPy does not yet support explicit multi-band clustering.

For all QNPy modules, your data must contain: `mjd` - MJD or time, `mag` - magnitude, and `magerr` - magnitude error.

Before running the script, you can manually create the following folders in the directory where your Python notebook is located:

1. `./preproc/` - for saving the transformed data
2. `./padded_lc/` - for saving the padded light curves
3. `./dataset/train/`, `./dataset/test/`, `./dataset/val/` - for organizing your dataset
4. `./output/prediction/test/data/`, `./output/prediction/test/plots/` - for organizing prediction results
5. `./output/prediction/train/data/`, `./output/prediction/train/plots/` - for organizing training results
6. `./output/prediction/val/data/`, `./output/prediction/val/plots/` - for organizing validation results

Modules and Their Functions
===========================

 **Clustering with SOM**

   **`Clustering_with_SOM.py`**

   In the clustering module, we first load the light curves from the directory. This also creates the ids from the file names. Thus, it is recommended to have the same light curves saved across all the different bands. Then, we pad the light curves to make them all the same length. In QNPy, we have seen that we require at least 100 data points for accurate modeling. Thus, we recommend that the light curves be padded to at least 100 points (even if the longest curve is under 100 points, which can be controlled through a keyword in the padding function). Finally, we scale the light curves. We have provided many different scalers including minmax, standard and robust scalers. Our `default` scaler is an adapted version of a minmax scaler that scales all the data to the range [-2,2].

   Then, a SOM is trained on the scaled data. The SOM contains different tunable hyperparameters to better adapt to different data sets. These hyperparameters can be tested with different metrics including quantization error, topographical error, silhouette score, davies-bouldin index, or calinski-harabasz score. The trained SOM can be saved as well.

   The trained SOM is then used to assign the IDs to different clusters. Then they can be saved into different folders.

   We also provide different plots for visualization of the data. These will be described in the plotting functions.

   Described below are the functions used in the module:

   .. code-block:: python

        def Plot_Lc(Light_Curve,header = 'Light Curve',save_fig = False,filename = 'Figure',x_axis = 'mjd',return_fig = False):
            '''
            Plots light curves interactively. Adapted from https://github.com/DamirBogdan39/time-series-analysis/tree/main
    
            Parameters
         ----------
         Light_Curve: Dataframe 
         The light curve to plot. Should be in a dataframe with mjd (or any x axis), mag and magerr
    
            header: str
            The header of the file
    
            save_fig: bool
            Whether to save the figure
    
            filename: str
            What to name the saved html file
    
            x_axis: str
            What to label the x axis 
    
            return_fig: bool
            Whether the figure is returned
    
            Returns
            --------
            Figure:
            The interactive plot of the light curve
            '''
    
   .. code-block:: python

        def Load_Light_Curves(folder,one_filter = True,filters = 'a',id_list = None):
            '''
            Loads light curves from a specified folder. Can be used to load either multiple filters or just one filter
    
            Parameters
            ----------
            folder: str 
            The folder where the light curves are stored
    
            one_filter: bool
            If set to true, the light curves are only stored in one folder without filters
    
            filters: list or str(if each filter is a single letter)
            The filters that are to be loaded. Each filter should have a subfolder named after it if there are more than one filters
    
            id_list: list of str or None
            The subset of IDs to load. If None, retrieves all files in the given folder. NOTE: make sure the ids are strings
    
            Returns
            --------
            light_curves: list of lists of dataframes
            The list of light curves arranged by filter
    
            ids: list
            The ids of the light curves (Ensure that they are the same in all filters)
            '''

   .. code-block:: python

        def Pad_Light_Curves(light_curves,minimum_length = 100,save_padded_lcs = False,padded_lcs_save_path = './',ids = None):
            '''
            Pads the light curves with the mean value at the end of the curve
    
            Parameters
            ----------
            light_curves: lists of dataframes 
            The light curves stored in a list
    
            minimum_length: int
            The minimum length to pad to
    
            save_padded_lcs: bool
            If True, will save the light curves into a folder known as Padded_Lc in the specified directory
    
            padded_lcs_save_path: str
          The directory to save the light curves in
    
            ids: list of str
            A list of the ids. Must provided in order to save the light curves
    
            Returns
            --------
            light_curves: list of lists
            The new padded light curves
            '''

   .. code-block:: python

        def scale_curves(light_curves,what_scaler = 'default',scale_times = True):
            '''
            Scaling the curves (from a single filter) from the choice of minmax, standard and robust. By default, it scales to a range of [-2,2]
            Parameters
            ----------
            light_curves: list of dataframes 
            The light curves stored in a list.
    
          what_scaler: string
            The type of scaler to use. There are default (see above), standard scaler, min-max scaler and robust scalers available
    
            scale_times: bool
            Whether to scale the time axis as well (These are always scaled to the default scaler)
    
            Returns
            --------
            scaled_curves: np.ndarray 
            The scaled light curves
    
            scaled_times:np.ndarray
            The scaled time steps. It is an empty list if the keyword scale_times is False
            '''

   .. code-block:: python

        def SOM_1D(scaled_curves,som_x = None,som_y = None,learning_rate = 0.1,sigma = 1.0,topology = 'rectangular',pca_init = True,\
        neighborhood_function='gaussian',train_mode = 'random',batch_size = 5,epochs = 50000,save_som = True,\
        model_save_path = './',random_seed = 21,stat = 'q',plot_frequency = 100,early_stopping_no = None):
            '''
            Training a SOM on ONE dimensional data (The magnitude of the light curves)
         Parameters
          ----------
         scaled_curves: list of dataframes 
         The scaled light curves stored in a list.
    
         som_x: int
         The x size of the SOM. If None is given, make sure the som_y is None as well. Then, it chooses the recommended SOM 
         size of sqrt(sqrt(length))
    
         som_y: int
         The y size of the SOM. If None is given, make sure the som_x is None as well. Then, it chooses the recommended SOM 
         size of sqrt(sqrt(length))
    
         learning_rate: float
         How much the SOM learns from the new data that it sees
    
         sigma: float
         The effect each node has on its neighboring nodes
    
            topology: 'rectangular' or 'hexagonal':
         The topology of the SOM. Note that visualizations are mainly built for the rectangular at the moment.
    
         pca_init: bool
          Whether to initialize the SOM weights randomly or to initialize by PCA of the input data
    
          neighborhood_function: str
          Choose from 'gaussian','mexican hat','bubble', or 'triangle'. These affect the influence of a node on its neighbors
    
          train_mode:'random' or 'all'
          When chosen random, it chooses a random curve each epoch. When trained on all, it batches the data and trains on every
           light curve for a certain number of epochs.
    
           batch_size: int
           How big the batch is for the 'all' train mode. The smaller the batch size, the finer the progress bar displayed
    
           epochs: int
          This is defined in two ways. If the train_mode is random, then it is the number of iterations that the SOM runs on.
          If it is all, then it is the number of times that the SOM trains on each input datapoint. Note that the lr and sigma
           decay in each epoch.
    
           save_som: bool
           Whether to save the trained SOM 
    
           model_save_path:str
           The file to save the SOM in
    
           random_seed:int
           The starting state of the random weights of the SOM. Use for reproducibility
    
           stat: 'q','t', or 'qt'
            Whether to record the quantization error,   topographical error or both. Note that calculating them is expensive
    
         plot_frequency: int
          The number of epochs
    
          early_stopping_no: int or None
          The number of batches to process before stopping. Use None if you should train on all
    
         Returns
         --------
         som_model:
         The trained SOM that can be saved or used for analysis
    
         q_error: list
          The quantization errors recorded
    
           t_error: list
           The topographic errors recorded
    
           indices_to_plot: list
           The indices to plot for the quantization or/and topographic errors
            '''

   .. code-block:: python 

        def plot_training(training_metric_results,metric,plotting_frequency,indices_to_plot,figsize = (10,10),save_figs = True,fig_save_path = './'):
            '''
         Plots the metric given (quantization error or topographic error)
    
          Parameters
         ----------
         training_metric_results: list 
         The result obtained from the SOM training
    
         metric: str
         Name of the metric
    
         plotting_frequency: int
          How much was the plotting frequency set during the SOM training
    
          indices_to_plot: list
          The indices to plot obtained from the SOM training
    
          figsize: tuple
          The size of the figure
    
          save_figs: bool
          Whether to save the figure or not
    
          fig_save_path:str
          Where to save the figure. Note that it creates a directory called Plots in the location given.
    
          Returns
          --------
          Plot:
          The plot of the metric
          '''

   .. code-block:: python

        def Plot_SOM_Scaled_Average(som_model,scaled_curves,dba = True,figsize = (10,10),save_figs = True,figs_save_path = './',\
        plot_weights = True,plot_avg = True,plot_background = True,one_fig = True,show_fig = True):
            '''
         Plotting the SOM Clusters with the average light curve and the SOM weights of each cluster. The average can be either simple mean
         or using a dba averaging method (https://github.com/fpetitjean/DBA)
    
         Parameters
          ----------
         som_model:  
          The trained SOM
    
         scaled_curves: np.ndarray
         The scaled curves that were the input for training
    
         dba: bool
         Whether to use Dynamic Barymetric Time Averaging
    
         figsize: tuple
         The size of the figure
    
         save_figs: bool
         Whether to save the figure or not
    
         fig_save_path: str
         Where to save the figure. Note that it creates a directory called Plots in the location given.
    
         plot_avg: bool
          Whether to plot the mean light curve of the cluster
    
          plot_weights: bool
          Whether to plot the SOM weight of the cluster
    
          plot_background: bool
          Whether to plot the light curves that make up the cluster
    
         one_fig: bool
          Whether to plot all the clusters into one figure or seperate figures
    
          show_fig: bool
          Whether to show each of the plots in the seperate figures case
    
         Returns
          --------
           Plot:
           The plots of the clusters
            '''

   .. code-block:: python

        def SOM_Distance_Map(som_model,figsize = (5,5),cmap = 'YlOrRd',save_figs = False,figs_save_path = './'):
            '''
            Plots a heatmap of the SOM Nodes. The brighter, the further away they are from their neighbors
    
            Parameters
            ----------
         som_model:  
         The trained SOM
    
            cmap: str
            The matplotlib based color scale to use for the plots
    
            figsize: tuple
         The size of the figure
    
            save_figs: bool
            Whether to save the figure or not
    
            fig_save_path: str
            Where to save the figure. Note that it creates a directory called Plots in the location given.
    
            Returns
            --------
            Plot:
            The heatmap plot
            '''

   .. code-block:: python

        def SOM_Activation_Map(som_model,figsize = (5,5),cmap = 'YlOrRd',save_figs = False,figs_save_path = './'):
            '''
            Plots a heatmap of the SOM Nodes. The brighter, the more light curves activate the SOM
    
            Parameters
            ----------
            som_model:  
            The trained SOM
    
            cmap: str
            The matplotlib based color scale to use for the plots
    
            figsize: tuple
            The size of the figure
    
            save_figs: bool
            Whether to save the figure or not
    
            fig_save_path: str
          Where to save the figure. Note that it creates a directory called Plots in the location given.
    
         Returns
         --------
            Plot:
         The heatmap plot
            '''

   .. code-block:: python

        def Assign_Cluster_Labels(som_model,scaled_curves,ids):
            '''
         Assigns Cluster labels to each of the curves, making a dataframe with their ids
    
            Parameters
            ----------
            som_model:  
         The trained SOM
    
            scaled_curves: np.ndarray
            The scaled curves that were used to train the SOM
    
            ids: list
         The ids of the curves
    
            Returns
            --------
         cluster_df: Dataframe
         A map matching each of the cluster ids with the cluster they belong to
            '''

   .. code-block:: python

        def SOM_Clusters_Histogram(cluster_map,color = 'tab:blue',save_figs = True,figs_save_path = './',figsize = (5,5)):
         '''
            Plots a heatmap of the SOM Nodes. The brighter, the further away they are from their neighbors
    
            Parameters
            ----------
            cluster_map:  
            The dataframe with each id and the cluster that it belongs to
    
            color: str
            The color to plot the histogram
    
         save_figs: bool
         Whether to save the figure or not
    
         fig_save_path: str
         Where to save the figure. Note that it creates a directory called Plots in the location given.
    
            figsize: tuple
         The size of the figure
    
         Returns
         --------
         Plot:
            The Histogram of how many curves are in each cluster
         '''

   .. code-block:: python

        def findMin(x, y, umat):
            '''
            Finds the minimum node in the unified matrix when given the x and y coordinate
    
            Parameters
            ----------
            x: int  
            The x position of the given input node
    
            y: int  
            The y position of the given input node
    
            umat: np.ndarry
            The unified distance matrix of the nodes of the SOM
    
            Returns
         --------
         minx, miny:
         The minumum x node and minimum y node
            '''

   .. code-block:: python

        def findInternalNode(x, y, umat):
            '''
         Finds the minimum node in the unified matrix when given the x and y coordinate, taking into account if the current node is min
    
            Parameters
            ----------
         x: int  
            The x position of the given input node
    
            y: int  
         The y position of the given input node
    
            umat: np.ndarry
            The unified distance matrix of the nodes of the SOM
    
            Returns
            --------
         minx, miny:
         The minumum x node and minimum y node
            '''

   .. code-block:: python

        def Get_Gradient_Cluster(som):
            '''
            Finds the center of the gradient for each node of the SOM
    
            Parameters
            ----------
            som: int  
            The trained SOM
    
            Returns
            --------
            cluster_centers:
            The center nodes that become the new gradient clusters
    
            cluster_pos:
            The original SOM cluster centers
            '''

   .. code-block:: python

        def Normal_Cluster_to_Grad(cluster_map,gradient_cluster_map):
            '''
            Maps the normal cluster map to the gradient clusters
    
            Parameters
            ----------
            cluster_map: pd.DataFrame  
            The map of the ids to the original SOM node clusters
    
            gradient_cluster_map: pd.DataFrame  
            The map of the ids to the gradient SOM node clusters
    
            Returns
            --------
            joint_map:
            Mapping of each SOM node cluster to the gradient cluster
            '''

   .. code-block:: python

        def Gradient_Cluster_Map(som,scaled_curves,ids,dimension = '1D',fill = 'mean',interpolation_kind = 'cubic',clusters = None,som_x = None,som_y = None):
            '''
            Translates the SOM nodes into larger clusters based on their gradients. Implementation of 
            https://homepage.cs.uri.edu/faculty/hamel/pubs/improved-umat-dmin11.pdf
    
            Parameters
            ----------
            som_model:  
            The trained SOM
    
            scaled_curves: np.ndarray
            The scaled curves used to train the SOM
    
            ids: list
            The ids of the curves
    
            dimension: str
            If 1D, does 1D clustering, else multivariate
    
            fill: str
            'mean' or 'interpolate'. Either the empty values are filled with the mean or they are interpolated with a function
    
            interpolation_kind: 
            Any of the scipy.interp1d interpolation kinds. Recommended to use cubic
    
            clusters:
            The clusters that the ids are in (only for multi-variate)
 
            som_x: int
            The x-dimensions of the SOM
    
            som_y: int
            The y-dimensions of the SOM
 
            Returns
            --------
            cluster_map:
            The new clusters that the ids are in
            '''

   .. code-block:: python

        def matplotlib_cmap_to_plotly(cmap, entries):
            '''
            Creates a colorscale used to create an interactive plot
    
            Parameters
            ----------
            cmap:  
            The colormap
    
            entries: 
            The colormap entries

            Returns
            --------
            colorscale:
            The colorscale for the interactive plot
            '''

   .. code-block:: python

        def plotStarburstMap(som):
            '''
            Interactive plot of the distance map and gradients of the SOM
    
            Parameters
            ----------
            som:  
            The trained SOM

            Returns
            --------
            Plot of the distance map and gradients
            '''

   .. code-block:: python

        def outliers_detection(clusters_df,som,scaled_curves,ids,outlier_percentage = 0.2):
            '''
            Gives the percentage of the clusters that have high quanitization errors (defined by percentile) for each cluster
    
            Parameters
            ----------
            clusters_df:
            A map of each of the ids to the clusters
    
            som:  
            The trained SOM
    
            scaled_curves: np.ndarray
            The scaled curves used to train the SOM
    
            ids: list
            The ids of the curves
    
            outlier_percentage: float
            This top percentile that defines an outlier
 
            Returns
            --------
            Plots:
            Distribution of Outliers per cluster and distribution of quantization error
            '''

   .. code-block:: python

        def Cluster_Metrics(scaled_curves,cluster_map,metric = 'Silhoutte'):
            '''
            Measures metrics related to the clustering
    
            Parameters
            ----------
            scaled_curves: np.ndarray
            The scaled curves used to train the SOM
    
            cluster_map: pd.Dataframe
            A map of each of the ids to the clusters
    
            metric: str
            The metric to be measured. It can be Silhoutte, DBI or CH. This is for silhoutte score, Davies-Bouldin index and calinski-harabasz score
 
            Returns
            --------
            score:
            The metric that is calculated
            '''

   .. code-block:: python

        def save_chosen_cluster(chosen_cluster,cluster_map,one_filter = True,filters = 'a',overwrite = True,save_path = './',source_path = './Light_Curves'):
            '''
         Saves the chosen cluster into a folder
    
            Parameters
            ----------
         chosen_cluster: int
            The cluster to save
    
            cluster_map: pd.Dataframe
            A map of each of the ids to the clusters
    
            one_filter: bool
            Whether to save just one filter or all the filters
    
            filters: str
         The filters to save
    
            overwrite: bool
            Whether to overwrite the current folder
    
            save_path: str
            The path to save to. This creates a folder for the cluster in that directory
    
            source_path: str
         The path that the light curves are saved in. If multifilter, provide the entire larger folder.
            '''

   .. code-block:: python

       def scale_to_range(series, min_val=-2, max_val=2):
            '''
            Scales a series to a range
    
            Parameters
            ----------
            series: pd.Series 
            The series to scale
    
            min_val: int
            The minimum value to scale to
    
            max_val: int
            The maximum value to scale to

            Returns
            --------
            scaled_series:
            The scaled series between the max and min values
            '''

   .. code-block:: python

        def masked_euclidean_distance(data1, data2, mask):
            '''
            Calculates the masked euclidean distance between two arrays using a common mask
    
            Parameters
            ----------
            data1: np.ndarray
            The first array
    
            data2: np.ndarray
            The second array
    
            mask: np.ma.mask
            The mask used to get the distance

            Returns
            --------
            masked_distance:
         The masked distance measure
            '''

   .. code-block: python

        def multi_band_processing(light_curves,ids,filter_names = 'ugriz',return_wide = False):
            '''
            Processes the light curves into a wide table
    
            Parameters
            ----------
            light_curves: 
            The light curves to be used
    
            ids: list,array
            The ids of the quasars
    
            filter_names: list or str(if the filters are one letter)
            The filters that are used
    
            return_wide: bool
            Whether to return the wide table or a flat table
    
            Returns
            --------
            light_curves_wide:
         The pivot table of the light curves with time steps
    
            light_curves_wide.isna():
            The mask used with the wide light curves
    
         OR 
    
            light_curves_flat:
         The flattened pivot table of the light curves
   
            mask_flat:
            The mask used
            '''

   .. code-block:: python

        def multi_band_clustering(light_curves,ids,filter_names = 'ugriz',som_x = None,som_y = None,sigma = 1.0,learning_rate = 0.5,\
        num_iterations = 2,batch_size = 5,early_stopping_no = None):
            '''
            Multiband light curve clustering
    
            Parameters
            ----------
            light_curves: 
            The light curves to be used
    
            ids: list,array
            The ids of the quasars
    
            filter_names: list or str(if the filters are one letter)
            The filters that are used
    
            som_x: int
            The x size of the SOM. If None is given, make sure the som_y is None as well. Then, it chooses the recommended SOM 
            size of sqrt(sqrt(length))
    
            som_y: int
            The y size of the SOM. If None is given, make sure the som_x is None as well. Then, it chooses the recommended SOM 
            size of sqrt(sqrt(length))
    
            sigma: float
            The effect each node has on its neighboring nodes
    
            learning_rate: float
            How much the SOM learns from the new data that it sees
    
            num_iterations: int
            The number of iterations that the som is trained on each batch
    
         batch_size: int
            The size of each batch
    
            early_stopping_no: int or None
            The number of batches to process before stopping. Use None if you should train on all
    
            Returns
            --------
            som:
            The trained SOM
    
            processed_light_curve:
            The flat light curves used for the SOM
    
            processed_mask:
            The mask used for the SOM
            '''

   .. code-block:: python

        def find_cluster_and_quantization_errors(som,data,masks):
            '''
            Finding the clusters and the quantization errors from the trained 2D SOM
    
            Parameters
         ----------
            som: 
            The trained SOM
    
            data: 
            The processed light curves from the trained SOM
    
            masks: 
            The masks used from the trained SOM
    
            Returns
            --------
            min_clusters:
            The clusters for each of the data points
    
            quantization_error:
            The quantization error of each of the data points
            '''

   .. code-block:: python

        def Get_Gradient_Cluster_2D(som,fill = 'mean',interpolation_kind = 'cubic'):
            '''
            Finding the gradient clusters from the 2D SOM
    
            Parameters
            ----------
            som: 
            The trained SOM
    
            fill: str
            'mean' or 'interpolate'. Either the empty values are filled with the mean or they are interpolated with a function
    
            interpolation_kind: 
            Any of the scipy.interp1d interpolation kinds. Recommended to use cubic
    
            Returns
            --------
            cluster_centers:
            The cluster centers
    
            cluster_pos:
            The cluster positions
            '''

   .. code-block:: python

        def create_dir_save_plot(path,plot_name):
            '''
            If there is no folder named plots in the path, it creates one and saves an figure
    
            Parameters
            ----------
            path: str  
            The path to create the Plots folder in
    
            plot_name: 
            The name to save the plot under
            ''' 

   .. code-block:: python

        def tolerant_mean(arrs):
            '''
            Calculates the mean of arrays without them having to be the same length
    
            Parameters
            ----------
            arrs:  
            The arrays to calculate the mean for 

            Returns
            --------
            mean:
            The tolerant mean of the arrays
    
            std:
            The tolerant std of the arrays
            '''

   .. code-block:: python

        def get_best_grid(number_of_objects):
            '''
            Creates a grid that is optimal for the number of objects
    
            Parameters
            ----------
            number_of_objects: int  
            The number of objects to make the grid for

            Returns
            --------
            rows:
            The number of rows for the grid
    
            cols:
            The number of cols for the grid
            '''

   .. code-block:: python

        def Averaging_Clusters(chosen_cluster,cluster_map,lcs,plot = True,dba = True):
            '''
            Creating a representation of the chosen cluster with the light curves and the average light curve
    
            Parameters
            ----------
            chosen_cluster: int 
            The cluster of interest
    
            cluster_map: pd.Dataframe
            A map of each of the ids to the clusters
    
            lcs: list of list of pd.Dataframes
            The light curves (provide the input from just one filter)
    
            plot: bool
            Whether to plot or just return the average value 
    
            dba: bool
            Whether to use Dynamic Barymetric Time Averaging or to use a simple mean of the light curves
    
            Returns
            --------
            average_x:
            The x_axis (i.e timesteps) of the average light curve
    
            average_y:
            The y_axis (i.e magnitudes) of the average light curve
    
            x:
            The timesteps of all the light curves concatenated into one array
    
            y: 
            The magnitudes of all the light curves concatenated into one array
    
            len(x):
            The length of all the light curves
            '''

   .. code-block:: python

        def Plot_All_Clusters(cluster_map,lcs,color = 'tab:blue',dba = True,figsize = (10,10),save_figs = True,figs_save_path = './'):
            '''
            Plots all of the clusters on a magnitude plot with the average representation included
    
            Parameters
            ----------
            cluster_map: pd.Dataframe
            A map of each of the ids to the clusters
    
            lcs: list of list of pd.Dataframes
            The light curves (provide the input from just one filter)
    
            color: str
            The color to plot the averaged curve in
    
            dba: bool
            Whether to use Dynamic Barymetric Time Averaging or to use a simple mean of the light curves
    
            figsize: tuple
            The figure size
    
            save_figs: bool
            Whether to save the figure or not
    
            figs_save_path: str
            Where to save the figure. Note that it is saved under a directory called Plots in that directory.
            '''

   .. code-block:: python

        def get_redshifts(redshifts_map):
            '''
            Gets all the redshifts from a redshifts map  
    
            Parameters
            ----------
            redshifts_map:  pd.Dataframe
            The mapping of ids to redshifts 

            Returns
            --------
            redshifts:
            The list of redshifts
            '''

   .. code-block:: python

        def get_fvars(lcs):
            '''
            Calculates the variability function of the light curves
    
            Parameters
            ----------
            lcs:  List of pd.Dataframes
            The light curves of interest

            Returns
            --------
            fvars:
            The list of variability functions
            '''

   .. code-block:: python

        def get_luminosities_and_masses(lcs, redshifts_map, H0 = 67.4, Om0 = 0.315):
            '''
            Randomly samples the luminosity and masses of the quasar black holes assuming a given Hubble Constant, Omega_0, and redshift
    
            Parameters
            ----------
            lcs:  List of pd.Dataframes
            The light curves of interest
    
            redshifts_map: pd.DataFrame
            The map from the ids to their redshifts
    
            H0: float
            The hubble constant at z=0
    
            Om0: float
            Omega matter: density of non-relativistic matter in units of the critical density at z=0. 

            Returns
            --------
            Log_lum:
            The logarithm of the luminosities
    
            Log_Mass:
            The logarithm of the masses
            '''

   .. code-block:: python

        def Cluster_Properties(cluster_map,selected_cluster,lcs,redshifts_map = None,plot = True,return_values = False,\
        the_property = 'all',save_figs = True,figs_save_path = './'):
            '''
            Getting the selected property of a chosen cluster
    
            Parameters
            ----------
            cluster_map: pd.Dataframe
            A map of each of the ids to the clusters
    
            chosen_cluster: int 
            The cluster of interest
    
            lcs: list of list of pd.Dataframes
            The light curves (provide the input from just one filter)
    
            redshifts_map: pd.Dataframe
            The redshift associated with each source id
    
            plot: bool
            Whether to plot or just return the average value 
    
            return_values: bool
            Whether to return the values for the property
    
            the_property: str
            The property to plot. Choice from z (redshift), Fvar (the variability function), Lum (luminosity), Mass, or all
    
            save_figs: bool
            Whether to save the figure or not
    
            figs_save_path: str
            Where to save the figure. Note that it is saved under a directory called Plots in that directory.
    
            Returns
            --------
            return_list: 
            The list of the property of interest
            '''

   .. code-block:: python

        def Cluster_Properties_Comparison(cluster_map,lcs,redshifts_map,the_property = 'Fvar',color = '#1f77b4',\
        figsize = (15,15),save_figs = True,figs_save_path = './'):
            '''
            Plotting the property of interest for all the clusters onto one figure
    
            Parameters
            ----------
            cluster_map: pd.Dataframe
            A map of each of the ids to the clusters
    
            lcs: list of list of pd.Dataframes
            The light curves (provide the input from just one filter)
    
            redshifts_map: pd.Dataframe
            The redshift associated with each source id
    
            the_property: str
            The property to plot. Choice from z (redshift), Fvar (the variability function), Lum (luminosity), Mass, or all
    
            color: str
            The color to make the histogram
    
            figsize: tuple
            The figure size
    
            save_figs: bool
            Whether to save the figure or not
    
            figs_save_path: str
            Where to save the figure. Note that it is saved under a directory called Plots in that directory.
    
            Returns
            --------
            return_list: 
            The list of the property of interest
            '''

   .. code-block:: python

        def SFplus(magnitudes):
            '''
            Calculates the S+ function of given light curves. S+ is the variance of magnitudes where the brightness increases
    
            Parameters
            ----------
            lcs:  List of pd.Dataframes
            The light curves of interest

            Returns
            --------
            sfplus:
            The list of S+
            '''

   .. code-block:: python

        def SFminus(magnitudes):
            '''
            Calculates the S- function of given light curves. S- is the variance of magnitudes where the brightness decreases
    
            Parameters
            ----------
            lcs:  List of pd.Dataframes
            The light curves of interest

            Returns
            --------
            sfminus:
            The list of S-
            '''

   .. code-block:: python

        def SF(magnitudes):
            '''
            Calculates the S function of given light curves. S is the variance of all magnitudes
    
            Parameters
            ----------
            lcs:  List of pd.Dataframes
            The light curves of interest

            Returns
            --------
            sf:
            The list of SFs
            '''

   .. code-block:: python

        def Structure_Function(cluster_map,selected_cluster,lcs,bins,save_figs = True,figs_save_path = './'):
            '''
            Create the structure function for a given cluster
    
            Parameters
            ----------
            cluster_map: pd.Dataframe
            A map of each of the ids to the clusters
    
            selected_cluster: int
            The cluster of interest
    
            lcs: list of list of pd.Dataframes
            The light curves (provide the input from just one filter)
    
            bins:int or list
            The bins to use for the structure function
    
            save_figs: bool
            Whether to save the figure or not
    
            figs_save_path: str
            Where to save the figure. Note that it is saved under a directory called Plots in that directory.
    
            Returns
            --------
            S+ and S- Plot: 
            A plot of the S+ and S- functions for the cluster
    
            Difference Plot:
            The evolution of the normalized S+ - S- throughout the observation time of the cluster
    
            S Plot:
            The evolution of the (regular) structure function through the observation time of the cluster
            '''

 **Preprocessing the Data**

   **`Preprocess.py`**

   In this module, we transform data in the range [-2,2]x[-2,2] to make training faster. This module also contains a function for padding light curves. This function does not need to be used if your light curves have 100 or more points. It is important to emphasize that all light curves must have the same number of points in order to train the model correctly. By testing the package, we found that it is best to do backward padding with the last measured value up to 100 points. You use the padded light curves to train the model, and later, for prediction and plotting, these points are subtracted. According to the needs of the user, another method for padding can be done. The data transformation function in this module also creates subsets of the data named your_curve_name plus and your_curve_name_minus. These subsets are made with respect to errors in magnitudes and serve to augment the model's training set. The original curves are saved as the name_of_your_curve_original.

   Before running this script, you must create the following folders in the directory where your Python notebook is located:

   1. `./preproc/` - It is going to be the folder for saving the transformed data
   2. `./padded_lc/` - It is going to be the folder for saving the padded light curves

   Your data must contain: `mjd` - MJD or time, `mag` - magnitude, and `magerr` - magnitude error

   .. code-block:: python

       def backward_pad_curves(folder_path, output_folder, desired_observations=100):
           """
           Backward padding the light curves with the last observed value for mag and magerr.
           If your data contains 'time' values it'll add +1 for padded values,
           and if your data contains 'MJD' values it will add +0.2

           :param str folder_path: The path to a folder containing the .csv files.
           :param str output_path: The path to a folder for saving the padded lc.
           :param int desired_observations: The number of points that our package is demanding is 100 but it can be more.

           :return: The padded light curves.
           :rtype: object

           How to use:
           padding = backward_pad_curves('./light_curves', './padded_lc', desired_observations=100)
           """

   .. code-block:: python

       def transform(data):
           """
           Transforming data into [-2,2]x[-2,2] range. This function needs to be uploaded before using it.

           :param data: Your data must contain: MJD or time, mag-magnitude, and magerr-magnitude error.
           :type data: object
           """

   .. code-block:: python

       def transform_and_save(files, DATA_SRC, DATA_DST, transform):
           """
           Transforms and saves a list of CSV files. The function also saves tr coefficients as a pickle file named trcoeff.pickle.

           :param list files: A list of CSV or TXT file names.
           :param str DATA_SRC: The path to the folder containing the CSV or TXT files.
           :param str DATA_DST: The path to the folder where the transformed CSV or TXT files will be saved.
           :param function transform: The transformation function defined previously.

           :return: A list of transformation coefficients for each file, where each element is a list containing the file name and the transformation coefficients Ax, Bx, Ay, and By.
           :rtype: list

           How to use:
           number_of_points, trcoeff = transform_and_save(files, DATA_SRC, DATA_DST, transform)
           """

 2. **SPLITTING AND TRAINING THE DATA**

   **`SPLITTING_AND_TRAINING.py`**

   We use this module to split the data into three subsamples that will serve as a test sample, a sample for model training, and a validation sample. This module also contains functions for training and saving models. It contains the following functions that must be executed in order.

   Before running this script, you must create the following folders in the directory where your Python notebook is located:

   1. `./dataset/train` - folder for saving data for training after splitting your original dataset
   2. `./dataset/test` - folder for test data 
   3. `./dataset/val` - folder for validation data
   4. `./output` - folder where you are going to save your trained model

   .. code-block:: python

       def create_split_folders(train_folder='./dataset/train/', test_folder='./dataset/test/', val_folder='./dataset/val/'):
            """
            Creates a TRAIN, TEST, and VAL folders in the directory.

            :param str train_folder: Path for saving the train data.
            :param str test_folder: Path for test data.
            :param str val_folder: Path for validation data.

            How to use: create_split_folders(train_folder='./dataset/train/', test_folder='./dataset/test/', val_folder='./dataset/val/')
            """

   .. code-block:: python

       def split_data(files, DATA_SRC, TRAIN_FOLDER, TEST_FOLDER, VAL_FOLDER):
            """
            Splits the data into TRAIN, TEST, and VAL folders.

            :param list files: A list of CSV file names.
            :param str DATA_SRC: Path to preprocessed data.
            :param str TRAIN_FOLDER: Path for saving the train data.
            :param str TEST_FOLDER: Path for saving the test data.
            :param str VAL_FOLDER: Path for saving the validation data.

            How to use: split_data(files, DATA_SRC, TRAIN_FOLDER, TEST_FOLDER, VAL_FOLDER)
            """

   **`TRAINING`** 

   Special note for mac os users: 

   When creating folders with mac operating systems, hidden `.DS_Store` files may be created. The user must delete these files before starting training from each folder. The best way is to go into each folder individually and run the command:

   .. code-block:: python

         !rm -f .DS_Store

   Important note: Deleting files using the `delete` directly in the folders does not remove hidden files.

   Before running the training function you must define:

   .. code-block:: python

         DATA_PATH_TRAIN = "./dataset/train" - path to train folder
         DATA_PATH_VAL = "./dataset/val" - path to val folder

         MODEL_PATH = "./output/cnp_model.pth" - folder for saving model

         BATCH_SIZE = 32 - training batch size MUST REMAIN 32
         EPOCHS = 6000 - This is optional
         EARLY_STOPPING_LIMIT = 3000 - This is optional
      
   .. code-block:: python

         def get_data_loader(data_path_train, data_path_val, batch_size):
          """
          
          --- Defining train and validation loader for training process and validation


            Args:
            :param str data_path_train: path to train folder
            :param str data_path_val: path to val folder
            :param batch_size: it is recommended to be 32

            How to use: trainLoader, valLoader = get_data_loader(DATA_PATH_TRAIN,BATCH SIZE)
          """

   .. code-block:: python

         def create_model_and_optimizer():
          """
            --Defines the model as Deterministic Model, optimizer as torch optimizer, criterion as LogProbLoss, mseMetric as MSELoss and maeMetric as MAELoss

            How to use: model, optimizer, criterion, mseMetric, maeMetric = create_model_and_optimizer(device)
             Device has to be defined before and it can be cuda or cpu
          """

   .. code-block:: python

         def train_model(model, train_loader, val_loader,criterion, optimizer, num_runs, epochs, early_stopping_limit, mse_metric, maeMetric, device):
          """
          -- Trains the model


          Args:
          model: Deterministic model
          train_loader: train loader
          val_loader: validation loader
          criterion: criterion
          optimizer: torch optimizer
          num_runs: The number of trainings 
          epochs: epochs for training. This is optional, but minimum of 3000 is recomended
          early_stopping_limit: limits the epochs for stopping the training. This is optional but minimum of 1500 is recomended
          mse_metric: mse metric
          mae_metric: mae metric
          device: torch device cpu or cuda
 
           How to use: If you want to save history_loss_train, history_loss_val, history_mse_train and history_mse_val for plotting you train your model like:

          history_loss_train, history_loss_val, history_mse_train, history_mse_val, history_mae_train, history_mae_val, epoch_counter_train_loss, epoch_counter_train_mse, epoch_counter_train_mae, epoch_counter_val_loss, epoch_counter_val_mse, epoch_counter_val_mae = st.train_model(model, trainLoader, valLoader, criterion, optimizer, 1, 3000, 1500, mseMetric, maeMetric, device)

          """
   .. code-block:: python

         def save_lists_to_csv(file_names,lists):
          """

          --saving the histories to lists

   
          args:
          :param list file_names: A list of file names to be used for saving the data. Each file name corresponds to a specific data list that will be saved in CSV format.
          :param list lists: A list of lists containing the data to be saved. Each inner list represents a set of rows to be written to a CSV file.

          How to use: 
          # Define the file names for saving the lists
          file_names = ["history_loss_train.csv", "history_loss_val.csv", "history_mse_train.csv", "history_mse_val.csv","history_mae_train.csv", "history_mae_val.csv", "epoch_counter_train_loss.csv", "epoch_counter_train_mse.csv", "epoch_counter_train_mae.csv", "epoch_counter_val_loss.csv","epoch_counter_val_mse.csv", "epoch_counter_val_mae.csv"]

          # Define the lists
          lists = [history_loss_train, history_loss_val, history_mse_train, history_mse_val, history_mae_train,
          history_mae_val, epoch_counter_train_loss, epoch_counter_train_mse, epoch_counter_train_mae,
          epoch_counter_val_loss, epoch_counter_val_mse, epoch_counter_val_mae]

          save_list= save_lists_to_csv(file_names, lists)
          """
   .. code-block:: python

       def plot_loss(history_loss_train_file, history_loss_val_file, epoch_counter_train_loss_file):
          """

          -- plotting the history losses


          Args:
          returned data from test_model
          How to use: 
   
          history_loss_train_file = './history_loss_train.csv'  # Replace with the path to your history_loss_train CSV file
          history_loss_val_file = './history_loss_val.csv'  # Replace with the path to your history_loss_val CSV file
          epoch_counter_train_loss_file = './epoch_counter_train_loss.csv'  # Replace with the path to your epoch_counter_train_loss CSV file
   
          logprobloss=plot_loss(history_loss_train_file, history_loss_val_file, epoch_counter_train_loss_file)

          """

   .. code-block:: python

       def plot_mse_metric(history_mse_train_file, history_mse_val_file, epoch_counter_train_mse_file):
          """

          -- plotting the mse metric


           args:
           returned data from test_model
           How to use: 
   
          history_mse_train_file = './history_mse_train.csv'  # Replace with the path to your history_mse_train CSV file
          history_mse_val_file = './history_mse_val.csv'  # Replace with the path to your history_mse_val CSV file
          epoch_counter_train_mse_file = './epoch_counter_train_mse.csv'  # Replace with the path to your epoch_counter_train_mse CSV file
   
          msemetric=plot_mse(history_mse_train_file, history_mse_val_file, epoch_counter_train_mse_file)

          """

   .. code-block:: python

       def plot_mae_metric(history_mae_train_file, history_mae_val_file, epoch_counter_train_mae_file):
          """

          -- plotting the mae metric


            args:
          returned data from test_model
          How to use: 
   
          history_mae_train_file = './history_mae_train.csv'  # Replace with the path to your history_mae_train CSV file
          history_mae_val_file = './history_mae_val.csv'  # Replace with the path to your history_mae_val CSV file
          epoch_counter_train_mae_file = './epoch_counter_train_mae.csv'  # Replace with the path to your epoch_counter_train_mae CSV file
   
          maemetric=plot_mae(history_mae_train_file, history_mae_val_file, epoch_counter_train_mae_file)
          """
   .. code-block:: python

       def save_model(model, MODEL_PATH):
          """

          -- saving the model


          Args:
          model: Deterministic model
          :param str MODEL_PATH: output path for saving the model

          How to use: save_model(model, MODEL_PATH)
          """

 3. **PREDICTION AND PLOTTING THE TRANSFORMED DATA, EACH CURVE INDIVIDUALLY**

  **`PREDICTION.py`**

  We use this module for prediction and plotting of models of transformed data. Each curve will be plotted separately. It contains the following functions that must be executed in order.

  Before running this script, you must create the following folders in the directory where your Python notebook is located:

  1. `./output/predictions/train/plots` - folder for saving training plots
  2. `./output/predictions/test/plots` - folder for saving test plots  
  3. `./output/predictions/val/plots` - folder for saving validation plots
  4. `./output/predictions/train/data` - folder for sving train data
  5. `./output/predictions/test/data` - folder for saving test data
  6. `./output/predictions/val/data` - folder for saving val data

   .. code-block:: python

       def prepare_output_dir(OUTPUT_PATH):
          """ 

          -- the function prepare_output_dir takes the `OUTPUT_PATH` as an argument and removes all files in the output directory using os.walk method.


          Args:
          :param str OUTPUT_PATH: path to output folder

          How to use: prepare_output_dir(OUTPUT_PATH)
          """

   .. code-block:: python

       def load_trained_model(MODEL_PATH, device):
          """ 

          --Uploading trained model


          agrs:
          :param str MODEL_PATH = path to model directorium
          :param device = torch device CPU or CUDA
          How to use: model=load_trained_model(MODEL_PATH, device)
          """

   .. code-block:: python

       def get_criteria():
          """

          -- Gives the criterion and mse_metric

          How to use: criterion, mseMetric=get_criteria()
          """

   .. code-block:: python

       def remove_padded_values_and_filter(folder_path):
          """

          -- Preparing data for plotting. It'll remove the padded values from lc and it'll delete artifitially added lc with plus and minus errors. If your lc are not padded it'll only delete additional curves


          Args:
          :param str folder_path: Path to folder where the curves are. In this case it'll be './dataset/test' or './dataset/train' or './dataset/val'

          How to use: 
          if __name__ == "__main__":
          folder_path = "./dataset/test"  # Change this to your dataset folder

          remove_padded_values_and_filter(folder_path)
          """

   .. code-block:: python

       def plot_function(target_x, target_y, context_x, context_y, pred_y, var, target_test_x, lcName, save = False, flagval=0, isTrainData = None, notTrainData = None):
          """

          -- Defines the plots of the light curve data and predicted mean and variance, and it should be imported separately
         

          Args:
          :param context_x: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the x values of the context points.
          :param context_y: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the y values of the context points.
          :param target_x: Array of shape BATCH_SIZE x NUM_TARGET that contains the x values of the target points.
          :param target_y: Array of shape BATCH_SIZE x NUM_TARGET that contains the ground truth y values of the target points.
          :param target_test_x: Array of shape BATCH_SIZE x 400 that contains uniformly spread points across in [-2, 2] range.
          :param pred_y: Array of shape BATCH_SIZE x 400 that contains predictions across [-2, 2] range.
          :param var: An array of shape BATCH_SIZE x 400  that contains the variance of predictions at target_test_x points.
          """

   .. code-block:: python

       def load_test_data(data_path):
          """

          -- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

          Args:
          :param str data_path: path to Test data

          How to use: testLoader=load_test_data(DATA_PATH_TEST)

          """

   .. code-block:: python

       def load_train_data(data_path):
          """

          -- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

          Args:
          :param str data_path: path to train data

          How to use: trainLoader=load_train_data(DATA_PATH_TRAIN)

          """

   .. code-block:: python

       def load_val_data(data_path):
          """

          -- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

          Args:
          :param str data_path: path to VAL data

          How to use: valLoader=load_val_data(DATA_PATH_VAL)

          """

   .. code-block:: python

       def plot_light_curves_from_test_set(model, testLoader, criterion, mseMetric, plot_function, device):
          """

          -- Ploting the transformed light curves from test set


          Args:
          :param model: Deterministic model
          :param testLoader: Uploaded test data
          :param criterion: criterion
          :param mseMetric: Mse Metric
          :param plot_function: plot function defined above
          :param device: torch device CPU or CUDA

          How to use: testMetrics = plot_light_curves_from_test_set(model, testLoader, criterion, mseMetric, plot_function, device)
          """

   .. code-block:: python

       def save_test_metrics(OUTPUT_PATH, testMetrics):
          """

          -- saving the test metrics as json file


          Args:
          :param str OUTPUT_PATH: path to output folder
          :param testMetrics: returned data from ploting function

           How to use: save_test_metrics(OUTPUT_PATH, testMetrics)
          """
   .. code-block:: python

       def plot_light_curves_from_train_set(model, trainLoader, criterion, mseMetric, plot_function, device):
          """

          -- Ploting the transformed light curves from train set


          Args:
          :param model: Deterministic model
          :param trainLoader: Uploaded trained data
          :param criterion: criterion
          :param mseMetric: Mse Metric
          :param plot_function: plot function defined above
          :param device: torch device CPU or CUDA

          How to use: trainMetrics = plot_light_curves_from_train_set(model, trainLoader, criterion, mseMetric, plot_function, device) 
          """

   .. code-block:: python

       def save_train_metrics(OUTPUT_PATH, testMetrics)
          """

          -- saving the train metrics as json file


          Args:
          :param str OUTPUT_PATH: path to output folder
          :param trainMetrics: returned data from ploting function

          How to use: save_train_metrics(OUTPUT_PATH, trainMetrics)
          """

   .. code-block:: python

       def plot_light_curves_from_val_set(model, valLoader, criterion, mseMetric, plot_function, device):
          """

          -- Ploting the transformed light curves from validation set


          Args:
          :param model: Deterministic model
          :param valLoader: Uploaded val data
          :param criterion: criterion
          :param mseMetric: Mse Metric
          :param plot_function: plot function defined above
          :param device: torch device CPU or CUDA

          How to use: valMetrics = plot_light_curves_from_val_set(model, valLoader, criterion, mseMetric, plot_function, device)
          """

   .. code-block:: python

       def save_val_metrics(OUTPUT_PATH, valMetrics):
          """

          -- saving the validation metrics as json file

          Args:
          :param str OUTPUT_PATH: path to output folder
          :param valMetrics: returned data from ploting function

          How to use: save_val_metrics(OUTPUT_PATH, valMetrics)
          """


 4. **PREDICTION AND PLOTTING THE TRANSFORMED DATA, IN ONE PDF FILE**

  **`PREDICTION_onePDF.py`**

  We use this module for prediction and plotting of models of transformed data. All curves will be plotted in one PDF file. This module contains the following functions that must be executed in order.

  Before running this script, you must create the following folders in the directory where your Python notebook is located:

  1. `./output/predictions/train/plots` -- folder for saving training plots
  2. `./output/predictions/test/plots` -- folder for saving test plots 
  3. `./output/predictions/val/plots` -- folder for saving validation plots
  4. `./output/predictions/train/data` -- folder for sving train data
  5. `./output/predictions/test/data` -- folder for saving test data
  6. `./output/predictions/val/data` -- folder for saving val data

   .. code-block:: python

       def clear_output_dir(output_path):
          """

          -- Removes all files in the specified output directory.
    

          Args:
          :param str output_path: The path to the output directory.

          How to use: clear_output_dir(OUTPUT_PATH)

          """

   .. code-block:: python

       def load_model(model_path, device):
          """
          --Loads a trained model from disk and moves it to the specified device.
    
          Args:
          :param str model_path: The path to the saved model.
          :param str or torch.device device: The device to load the model onto, CPU or CUDA

          How to use: model = load_model(MODEL_PATH, device)
          """

   .. code-block:: python

       def get_criteria():
          """

          -- Gives the criterion and mse_metric


          How to use: criterion, mseMetric=get_criteria()
          """

   .. code-block:: python

       def remove_padded_values_and_filter(folder_path):
          """

          -- Preparing data for plotting. It'll remove the padded values from lc and it'll delete artifitially added lc with plus and minus errors. If your lc are not padded it'll only delete additional curves


            Args:
          :param str folder_path: Path to folder where the curves are. In this case it'll be './dataset/test' or './dataset/train' or './dataset/val'

          How to use: 
          if __name__ == "__main__":
            folder_path = "./dataset/test"  # Change this to your dataset folder

          remove_padded_values_and_filter(folder_path)
          """

   .. code-block:: python
   
       dev plot_function(target_x, target_y, context_x, context_y, pred_y, var, target_test_x, lcName, save = False, flagval=0, isTrainData = None, notTrainData = None):
          """

          -- Defines the plots of the light curve data and predicted mean and variance
    

            Args:
          context_x: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the x values of the context points.
          context_y: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the y values of the context points.
          target_x: Array of shape BATCH_SIZE x NUM_TARGET that contains the x values of the target points.
          target_y: Array of shape BATCH_SIZE x NUM_TARGET that contains the ground truth y values of the target points.
          target_test_x: Array of shape BATCH_SIZE x 400 that contains uniformly spread points across in [-2, 2] range.
          pred_y: Array of shape BATCH_SIZE x 400 that contains predictions across [-2, 2] range.
          var: An array of shape BATCH_SIZE x 400  that contains the variance of predictions at target_test_x points.
          """

   .. code-block:: python

       def load_test_data(data_path):
          """

          -- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

          Args:
          :param str data_path: path to Test data

          How to use: testLoader=load_test_data(DATA_PATH_TEST)

          """

   .. code-block:: python

       def load_train_data(data_path):
          """

          -- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

          Args:
          :param str data_path: path to train data

          How to use: trainLoader=load_train_data(DATA_PATH_TRAIN)

          """

   .. code-block:: python

       def load_val_data(data_path):
          """

          -- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

          Args:
          :param str data_path: path to VAL data

          How to use: valLoader=load_val_data(DATA_PATH_VAL)

          """

   .. code-block:: python

       def plot_test_light_curves(model, testLoader, criterion, mseMetric, plot_function, device):
          """
          -- ploting the test set in range [-2,2]


          Args:
          :param model: model
          :param testLoader: Test set
          :param criterion: criterion
          :param mseMetric: mse Metric
          :param plot_function: defined above
          :param device: Torch device, CPU or CUDA


          how to use: testMetrics=plot_test_light_curves(model, testLoader, criterion, mseMetric, plot_function, out_pdf_test, device)
          """

   .. code-block:: python

       def save_test_metrics(OUTPUT_PATH, testMetrics):
          """

          -- saving the test metrics as json file


          Args:
          :param str OUTPUT_PATH: path to output folder
          :param testMetrics: returned data from ploting function

          How to use: save_test_metrics(OUTPUT_PATH, testMetrics)
          """

   .. code-block:: python

       def plot_train_light_curves(model, trainLoader, criterion, mseMetric, plot_function, device):
          """

          -- Ploting the transformed light curves from train set


          Args:
          :param model: Deterministic model
          :param trainLoader: Uploaded trained data
          :param criterion: criterion
          :param mseMetric: Mse Metric
          :param plot_function: plot function defined above
          :param device: torch device CPU or CUDA
      

          How to use: trainMetrics=plot_train_light_curves(model, trainLoader, criterion, mseMetric, plot_function, device)
          """

   .. code-block:: python

       save_train_metrics(OUTPUT_PATH, testMetrics)
          """

          -- saving the train metrics as json file


          Args:
          :param str OUTPUT_PATH: path to output folder
          :param trainMetrics: returned data from ploting function

          How to use: save_train_metrics(OUTPUT_PATH, trainMetrics)
          """

   .. code-block:: python

       def plot_val_light_curves(model, valLoader, criterion, mseMetric, plot_function, device):
          """

          -- Ploting the transformed light curves from val set


          Args:
          :param model: Deterministic model
          :param valLoader: Uploaded val data
          :param criterion: criterion
          :param mseMetric: Mse Metric
          :param plot_function: plot function defined above
          :param device: torch device CPU or CUDA

          How to use: valMetrics = plot_val_light_curves(model, valLoader, criterion, mseMetric, plot_function, device)
          """

   .. code-block:: python

       def save_val_metrics(OUTPUT_PATH, valMetrics):
          """

          -- saving the val metrics as json file


          Args:
          :param str OUTPUT_PATH: path to output folder
          :param valMetrics: returned data from ploting function

          How to use: save_val_metrics(OUTPUT_PATH, valMetrics)
          """


 5. **PREDICTION AND PLOTTING THE DATA IN ORIGINAL DATA RANGE, EACH CURVE INDIVIDUALLY**

   **`PREDICTION_Original_mjd.py`**

   We use this module to predict and plot the model in the original range of data. All curves are plotted individually. This module contains the following functions that must be executed in order.

   Before running this script, you must create the following folders in the directory where your Python notebook is located:

  1. `./output/predictions/train/plots` -- folder for saving training plots
  2. `./output/predictions/test/plots` -- folder for saving test plots 
  3. `./output/predictions/val/plots` -- folder for saving validation plots
  4. `./output/predictions/train/data` -- folder for sving train data
  5. `./output/predictions/test/data` -- folder for saving test data
  6. `./output/predictions/val/data` -- folder for saving val data

   .. code-block:: python

       def prepare_output_dir(OUTPUT_PATH):
          """

          -- the function prepare_output_dir takes the OUTPUT_PATH       as an argument and removes all files in the output    directory using os.walk method.


          Args:
          :param str OUTPUT_PATH: path to output folder

          How to use: prepare_output_dir(OUTPUT_PATH)
          """

   .. code-block:: python

       def load_trained_model(MODEL_PATH, device):
          """

          --Uploading trained model


            agrs:
          :param str MODEL_PATH = path to model directorium
          :param str device = torch device CPU or CUDA

          How to use: model=load_trained_model(MODEL_PATH, device)
          """

   .. code-block:: python

       def get_criteria():
          """

          -- Gives the criterion and mse_metric


          How to use: criterion, mseMetric=get_criteria()
          """

   .. code-block:: python

       def remove_padded_values_and_filter(folder_path):
          """

          -- Preparing data for plotting. It'll remove the padded values from lc and it'll delete artifitially added lc with plus and minus errors. If your lc are not padded it'll only delete additional curves


          Args:
          :param str folder_path: Path to folder where the curves are. In this case it'll be './dataset/test' or './dataset/train' or './dataset/val'

          How to use: 
          if __name__ == "__main__":
          folder_path = "./dataset/test"  # Change this to your dataset folder

          remove_padded_values_and_filter(folder_path)
          """

   .. code-block:: python

       def load_trcoeff():
          """ 

          -- loading the original coefficients from pickle file


          How to use: tr=load_trcoeff()

          """

   .. code-block:: python

       def plot_function2(tr,target_x, target_y, context_x, context_y, yerr1, pred_y, var, target_test_x, lcName, save = False, isTrainData = None, flagval = 0, notTrainData = None):
          """

          -- function for ploting the light curves


          context_x: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the x values of the context points.
          context_y: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the y values of the context points.
          target_x: Array of shape BATCH_SIZE x NUM_TARGET that contains the x values of the target points.
          target_y: Array of shape BATCH_SIZE x NUM_TARGET that contains the ground truth y values of the target points.
          target_test_x: Array of shape BATCH_SIZE x 400 that contains uniformly spread points across in [-2, 2] range.
          yerr1: Array of shape BATCH_SIZE x NUM_measurement_error that contains the measurement errors.
          pred_y: Array of shape BATCH_SIZE x 400 that contains predictions across [-2, 2] range.
          var: An array of shape BATCH_SIZE x 400  that contains the variance of predictions at target_test_x points.
          tr: array of data in pickle format needed to backtransform data from [-2,2] x [-2,2] to MJD x Mag
          """

   .. code-block:: python

       def load_test_data(data_path):
          """

          -- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

          Args:
          :param str data_path: path to Test data

          How to use: testLoader=load_test_data(DATA_PATH_TEST)

          """

   .. code-block:: python

       def load_train_data(data_path):
          """

          -- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

          Args:
          :param str data_path: path to train data

           How to use: trainLoader=load_train_data(DATA_PATH_TRAIN)

          """

   .. code-block:: python

       def load_val_data(data_path):
          """

          -- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

          Args:
          :param str data_path: path to VAL data

           How to use: valLoader=load_val_data(DATA_PATH_VAL)

          """

   .. code-block:: python

       def plot_test_data(model, testLoader, criterion, mseMetric, plot_function, device, tr):
          """

          -- Ploting the light curves from test set in original mjd range


            Args:
          :param model: Deterministic model
          :param testLoader: Uploaded test data
          :param criterion: criterion
          :param mseMetric: Mse Metric
          :param plot_function: plot function defined above
          :param device: torch device CPU or CUDA
          :param tr: trcoeff from pickle file

          How  to use: testMetrics=plot_test_data(model, testLoader, criterion, mseMetric, plot_function2, device, tr)
          """

   .. code-block:: python

       def save_test_metrics(OUTPUT_PATH, testMetrics):
          """

          -- saving the test metrics as json file


          Args:
          :param str OUTPUT_PATH: path to output folder
          :param testMetrics: returned data from ploting function

          How to use: save_test_metrics(OUTPUT_PATH, testMetrics)
          """

   .. code-block:: python

       def plot_train_light_curves(trainLoader, model, criterion, mseMetric, plot_function, device, tr):
          """

          -- Ploting the light curves from train set in original mjd range


          Args:
          :param model: Deterministic model
          :param trainLoader: Uploaded trained data
          :param criterion: criterion
          :param mseMetric: Mse Metric
          :param plot_function: plot function defined above
          :param device: torch device CPU or CUDA
          :param tr: trcoeff from pickle file

          How to use: trainMetrics=plot_train_light_curves(trainLoader, model, criterion, mseMetric, plot_function2, device,tr)
          """

   .. code-block:: python

       def save_train_metrics(OUTPUT_PATH, testMetrics):
          """

          -- saving the train metrics as json file


           Args:
          :param str OUTPUT_PATH: path to output folder
          :param trainMetrics: returned data from ploting function

          How to use: save_train_metrics(OUTPUT_PATH, trainMetrics)
          """

   .. code-block:: python

       def plot_val_curves(model, valLoader, criterion, mseMetric, plot_function, device, tr):
          """

          -- Ploting the light curves from val set in original mjd range


          Args:
          :param model: Deterministic model
          :param valLoader: Uploaded val data
          :param criterion: criterion
          :param mseMetric: Mse Metric
          :param plot_function: plot function defined above
          :param device: torch device CPU or CUDA
          :param tr: trcoeff from pikle file

          How to use: valMetrics=plot_val_curves(model, valLoader, criterion, mseMetric, plot_function2, device,tr)
          """

   .. code-block:: python

       def save_val_metrics(OUTPUT_PATH, valMetrics):
          """

          -- saving the val metrics as json file


          Args:
          :param str OUTPUT_PATH: path to output folder
          valMetrics: returned data from ploting function

          How to use: save_val_metrics(OUTPUT_PATH, valMetrics)
          """   


 6. **PREDICTION AND PLOTTING THE DATA IN ORIGINAL DATA RANGE, IN ONE PDF FILE**

  **`PREDICTION_onePDF_original_mjd.py`**

  We use this module to predict and plot the model in the original range of data. All curves are plotted in one PDF file. This module contains the following functions that must be executed in order.

  Before running this script, you must create the following folders in the directory where your Python notebook is located:
  
  1. `./output/predictions/train/plots` -- folder for saving training plots
  2. `./output/predictions/test/plots` -- folder for saving test plots 
  3. `./output/predictions/val/plots` -- folder for saving validation plots
  4. `./output/predictions/train/data` -- folder for sving train data
  5. `./output/predictions/test/data` -- folder for saving test data
  6. `./output/predictions/val/data` -- folder for saving val data

   .. code-block:: python

       def clear_output_dir(output_path):
          """

          -- Removes all files in the specified output directory.
    

          Args:
          :param str output_path: The path to the output directory.

          How to use: clear_output_dir(OUTPUT_PATH)

          """

   .. code-block:: python

       def load_trained_model(MODEL_PATH, device):
          """

          --Uploading trained model


            agrs:
          :param str MODEL_PATH = path to model directorium
          :param str device = torch device CPU or CUDA

          How to use: model=load_trained_model(MODEL_PATH, device)
          """

   .. code-block:: python

       def get_criteria():
          """

          -- Gives the criterion and mse_metric


          How to use: criterion, mseMetric=get_criteria()
          """

   .. code-block:: python

       def remove_padded_values_and_filter(folder_path):
          """

          -- Preparing data for plotting. It'll remove the padded values from lc and it'll delete artifitially added lc with plus and minus errors. If your lc are not padded it'll only delete additional curves


          Args:
          :param str folder_path: Path to folder where the curves are. In this case it'll be './dataset/test' or './dataset/train' or './dataset/val'

          How to use: 
          if __name__ == "__main__":
          folder_path = "./dataset/test"  # Change this to your dataset folder

          remove_padded_values_and_filter(folder_path)
          """

   .. code-block:: python

       def load_trcoeff():
          """ 

          -- loading the original coefficients from pickle file


          How to use: tr=load_trcoeff()

          """

   .. code-block:: python

       def plot_function2(tr,target_x, target_y, context_x, context_y, yerr1, pred_y, var, target_test_x, lcName, save = False, isTrainData = None, flagval = 0, notTrainData = None):
          """

          -- function for ploting the light curves. It needs to be uploaded separately


          context_x: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the x values of the context points.
          context_y: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the y values of the context points.
          target_x: Array of shape BATCH_SIZE x NUM_TARGET that contains the x values of the target points.
          target_y: Array of shape BATCH_SIZE x NUM_TARGET that contains the ground truth y values of the target points.
          target_test_x: Array of shape BATCH_SIZE x 400 that contains uniformly spread points across in [-2, 2] range.
          yerr1: Array of shape BATCH_SIZE x NUM_measurement_error that contains the measurement errors.
          pred_y: Array of shape BATCH_SIZE x 400 that contains predictions across [-2, 2] range.
          var: An array of shape BATCH_SIZE x 400  that contains the variance of predictions at target_test_x points.
          tr: array of data in pickle format needed to backtransform data from [-2,2] x [-2,2] to MJD x Mag
          """

   .. code-block:: python

       def load_test_data(data_path):
          """

          -- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

          Args:
          :param str data_path: path to Test data

          How to use: testLoader=load_test_data(DATA_PATH_TEST)

          """

   .. code-block:: python

       def load_train_data(data_path):
          """

          -- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

          Args:
          :param str data_path: path to train data

           How to use: trainLoader=load_train_data(DATA_PATH_TRAIN)

          """

   .. code-block:: python

       def load_val_data(data_path):
          """

          -- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

          Args:
          :param str data_path: path to VAL data

           How to use: valLoader=load_val_data(DATA_PATH_VAL)

          """

   .. code-block:: python

       def plot_test_light_curves(model, testLoader, criterion, mseMetric, plot_function2, device,tr):
          """

          -- ploting the test set in original range


          Args:
          :param model: model
          :param testLoader: Test set
          :param criterion: criterion
          :param mseMetric: mse Metric
          :param plot_function2: defined above
          :param device: Torch device, CPU or CUDA
          :param tr: trcoeff from pickle file

          how to use: testMetrics=plot_test_light_curves(model, testLoader, criterion, mseMetric, plot_function2, device,tr)
          """

   .. code-block:: python

       def save_test_metrics(OUTPUT_PATH, testMetrics):
          """

          -- saving the test metrics as json file


          Args:
          :param str OUTPUT_PATH: path to output folder
           testMetrics: returned data from ploting function

          How to use: save_test_metrics(OUTPUT_PATH, testMetrics)
          """

   .. code-block:: python

       def plot_train_light_curves(model, trainLoader, criterion, mseMetric, plot_function2, device,tr):
          """

          -- Ploting the light curves from train set in original mjd range


          Args:
          :param model: Deterministic model
          :param trainLoader: Uploaded trained data
          :param criterion: criterion
          :param mseMetric: Mse Metric
          :param plot_function: plot function defined above
          :param device: torch device CPU or CUDA
          :param tr: trcoeff from pickle file

          How to use: trainMetrics=plot_train_light_curves(model, trainLoader, criterion, mseMetric, plot_function2, device,tr)
          """

   .. code-block:: python

       def save_train_metrics(OUTPUT_PATH, testMetrics):
          """

          -- saving the train metrics as json file


          Args:
          :param str OUTPUT_PATH: path to output folder
          :param trainMetrics: returned data from ploting function

          How to use: save_train_metrics(OUTPUT_PATH, trainMetrics)
          """

   .. code-block:: python

       def plot_val_light_curves(model, valLoader, criterion, mseMetric, plot_function2, device,tr):
          """

          -- Ploting the light curves from val set in original mjd range


          Args:
          :param model: Deterministic model
          :param valLoader: Uploaded val data
          :param criterion: criterion
          :param mseMetric: Mse Metric
          :param plot_function: plot function defined above
          :param device: torch device CPU or CUDA
          :param tr: trcoeff from pickle file

          How to use: valMetrics=plot_val_light_curves(model, valLoader, criterion, mseMetric, plot_function2, device,tr)
          """

   .. code-block:: python

       def save_val_metrics(OUTPUT_PATH, valMetrics):
          """

          -- saving the val metrics as json file


          Args:
          :param str OUTPUT_PATH: path to output folder
          :param valMetrics: returned data from ploting function

          How to use: save_val_metrics(OUTPUT_PATH, valMetrics)
          """

Future release
==============

When investigating large data sets, it turns out to be benneficial to do light curve clustering. For this application, we have decided to incorporate a data clustering module into version 0.0.2 of QNPy python package, which will work on the basis of the Self Organising Map (SOM) algorithm. Currently, this version of the package is undergoing extensive testing and its publication is expected in mid-2024.

Frequently Asked Questions
==========================

**Q** What should my input data look like?

A: The input data should have three columns: mjd - Modified Julian Date or time, mag - magnitude, and magerr - magnitude error

**Q** Do I need to run all four prediction modules?

A: No, it is enough to run only one prediction module, depending on what you want on the final plots. There are four options for prediction and plotting namely:

1. all curves are plotted separately and plots contain transformed data.
2. all curves are plotted in one pdf document and contain transformed data
3. all curves are plotted separately and the plots contain the original data.
4. all curves are plotted in one pdf document and contain original data

**Q** Can the package be used for other uses outside of astronomy?

A: Yes, the package can be used for different types of time series analysis.

Licence
=======

MIT License



Copyright (c) 2023 Marina Pavlovic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

