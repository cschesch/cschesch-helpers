import re
import numpy as np
import matplotlib.pyplot as plt
import stargazer

def Capitalize(string):
    """
    Capitalizes the first letter of a string.

    Parameters
    ----------
    string : str
        The string to capitalize.

    Returns
    -------
    str
        The string with the first letter capitalized.
    """
    return string[0].capitalize() + string[1:]

def clean_filename(filename):
    """
    Cleans up a filename by converting it to lowercase, replacing spaces with underscores,
    and removing non-word characters.

    Parameters
    ----------
    filename : str
        The filename to clean.

    Returns
    -------
    str
        The cleaned filename.
    """
    filename = filename.lower().replace(' ', '_')
    filename = filename.replace('/', '_over_')
    filename = re.sub(r'\W+', '', filename)
    filename = re.sub('_+', '_', filename)
    return filename

def generate_lower_order_interactions(vars_in_interaction):
    """
    Generate all lower-order interaction terms from a list of variables.

    Args:
        vars_in_interaction (list): A list of variables involved in an interaction.

    Returns:
        lower_order_interactions (list): A list of all lower-order interaction terms.
    """
    lower_order_interactions = []

    # For each index, generate interaction terms by leaving out one term at a time
    for i in range(len(vars_in_interaction)):
        lower_order = vars_in_interaction[:i] + vars_in_interaction[(i+1):]

        # Add interaction term to list
        if lower_order:  # only if there is something left
            lower_order_interactions.append(':'.join(lower_order))

        # If there are more than two variables left, generate lower-order interactions recursively
        if len(lower_order) > 1:
            lower_order_interactions.extend(generate_lower_order_interactions(lower_order))

    return lower_order_interactions


def generate_covariate_latexdict(latexdict, cov_names):
    """
    Generate a dictionary mapping covariate names to their latex-ready versions.

    Args:
        latexdict (dict): A dictionary mapping original variable names to their latex-ready versions.
        cov_names (list): A list of covariate names.

    Returns:
        cov_latexdict (dict): A dictionary mapping covariate names to their latex-ready versions.
    """
    # Initialize the output dictionary
    cov_latexdict = {}

    # For each covariate name
    for cov_name in cov_names:
        # Split the covariate name into its constituent variables
        vars_in_interaction = cov_name.split(':')

        # If the covariate is an interaction term
        if len(vars_in_interaction) > 1:
            # Generate all lower-order interactions
            lower_order_interactions = generate_lower_order_interactions(vars_in_interaction)
            # Check that all lower-order interactions are included
            assert set(lower_order_interactions).issubset(cov_names), \
                f"Not all lower-order interactions of '{cov_name}' are included. " \
                f"Missing interactions: {set(lower_order_interactions) - set(cov_names)}"
            # Sort vars_in_interaction based on the order of variables in latexdict.keys()
            correct_order_vars = sorted(vars_in_interaction, key=lambda var: -list(latexdict.keys()).index(var) if var in latexdict.keys() else np.inf)
            # If the order is not correct, raise an assertion error
            assert vars_in_interaction == correct_order_vars, \
                f"Variable order in '{cov_name}' does not match the order in latexdict. " \
                f"Current order: {':'.join(vars_in_interaction)}. Correct order: {':'.join(correct_order_vars)}."
            
        # Replace original variable names with their latex-ready versions
        clean_vars_in_interaction = [latexdict.get(var, var) for var in vars_in_interaction]
        # Join the clean variables with ' $\\times$ ' for interaction terms
        clean_cov_name = ' $\\times$ '.join(clean_vars_in_interaction)

        # Add the clean covariate name to the output dictionary
        cov_latexdict[cov_name] = clean_cov_name

    return cov_latexdict

def savereg(stargazer, filename, tablepath, latexdict, adjustwidth='auto', debug=False):
    """
    Function to save a stargazer-type table to a .tex file.

    Args:
        stargazer (object): Stargazer object.
        filename (str): Name of the file to save the table as.
        tablepath (str): Path where to save the table.
        adjustwidth (str, optional): Whether to adjust the table width, can be either 'auto' or a boolean. 
            If 'auto', adjust width if there are more than 4 models. Defaults to 'auto'.
        debug (bool, optional): If True, prints debugging information. Defaults to False.
    """
    
    # Cleans the filename using predefined helper function
    filename = clean_filename(filename)
    
    # Determines whether to adjust table width
    if adjustwidth == 'auto': 
        adjustwidth = (len(stargazer.models)>4)
    
    # Set parameters for the stargazer table
    stargazer.show_degrees_of_freedom(False)
    stargazer.show_model_numbers(False)
    stargazer.append_notes(False)
    stargazer.show_notes = False
    stargazer.show_adj_r2 = False
    stargazer.show_residual_std_err = False
    stargazer.show_f_statistic = False
    
    # Rename variables and interaction variables
    cov_latexdict = generate_covariate_latexdict(latexdict, stargazer.cov_names)
    stargazer.rename_covariates(cov_latexdict)

    # Overwrite R2 with pseudo-R2 if provided
    if any([hasattr(model, 'prsquasecond') for model in stargazer.models]):
        for model, model_data in zip(stargazer.models, stargazer.model_data):
            model_data['r2'] = model.prsquasecond                        
            if not "$R^2$ reports McFadden's pseudo-$R^2$." in stargazer.custom_notes:
                stargazer.add_custom_notes(stargazer.custom_notes+["$R^2$ reports McFadden's pseudo-$R^2$."])
    
    # Render table to LaTeX and do post-processing
    latex = stargazer.render_latex()
    latex = latex.replace('nan', '')
    latex = latex.replace('\cline{'+str(stargazer.num_models)+'-'+str(stargazer.num_models+1)+'}', '\cline{2-'+str(stargazer.num_models+1)+'}')
    if adjustwidth:
        latex = latex.replace('\\begin{tabular}', '\\begin{adjustbox}{width=\linewidth}\\begin{tabular}')
        latex = latex.replace('\end{tabular}', '\end{tabular}\end{adjustbox}')
    
    # Write the full table to a .tex file
    with open(tablepath+filename+'.tex', 'w') as f: 
        f.write(latex)
    if debug:
        print('Saved as: '+tablepath+filename+'.tex')

    # Write only the tabular part of the table to a .tex file
    latex_table = latex.partition("\\begin{tabular}")[1] + latex.partition("\\begin{tabular}")[2]
    latex_table = latex_table.partition("\end{tabular}")[0] + latex_table.partition("\end{tabular}")[1]
    with open(tablepath+filename+'_tabular.tex', 'w') as f: 
        f.write(latex_table)

def savefig(ax, filename, figurepath):
    """
    Function to save a figure to a .pdf and .png file.

    Args:
        ax (matplotlib.axes.Axes): Axes object representing the figure to save.
        filename (str): Name of the file to save the figure as.
        figurepath (str): The path to save the figure in.
    """
    # Cleans the filename using predefined helper function
    filename = clean_filename(filename)

    # Check if ax has title attribute and hide it for saving if so
    if hasattr(ax, 'title'):
        plt.setp(ax.title, visible=False)
        plt.savefig(figurepath+filename+'_notitle.pdf', bbox_inches='tight')
        plt.setp(ax.title, visible=True) # Make the title visible again
    # Check if ax has suptitle attribute and hide it for saving if so
    elif hasattr(ax, 'suptitle'):
        plt.setp(ax.title, visible=False)
        plt.savefig(figurepath+filename+'_notitle.pdf', bbox_inches='tight')
        plt.setp(ax.title, visible=True) # Make the title visible again

    # Save the figure with title as .pdf and .png
    plt.savefig(figurepath+filename+'.pdf', bbox_inches='tight')
    plt.savefig(figurepath+filename+'.png', facecolor='w', dpi=500, bbox_inches='tight')
    plt.show() # Show the plot
    print('Saved to: '+figurepath+filename+'.pdf')