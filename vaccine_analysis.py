# COVID-19 Vaccination Analysis in India
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

# Set visualization style
plt.style.use('ggplot')

# Load the dataset
print("Loading COVID-19 vaccination dataset...")
try:
    data = pd.read_csv("covid_vaccine_statewise.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file 'covid_vaccine_statewise.csv' not found.")
    print("Please download from: https://www.kaggle.com/sudalairajkumar/covid19-in-india?select=covid_vaccine_statewise.csv")
    exit(1)

# a. Describe the dataset
print("\n" + "="*70)
print("A. DATASET DESCRIPTION")
print("="*70)

# Basic dataset information
print(f"Dataset Shape: {data.shape[0]} rows and {data.shape[1]} columns")
print("\nColumn Names:")
for i, col in enumerate(data.columns, 1):
    print(f"{i}. {col}")

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Dataset summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values by Column:")
missing_values = data.isnull().sum()
missing_percent = (missing_values / len(data)) * 100
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percent
})
print(missing_data[missing_data['Missing Values'] > 0])

# Time period of the dataset
print("\nTime Period:")
try:
    data['Updated On'] = pd.to_datetime(data['Updated On'], dayfirst=True, errors='coerce')
    print(f"From {data['Updated On'].min().strftime('%d-%m-%Y')} to {data['Updated On'].max().strftime('%d-%m-%Y')}")
except:
    print("Date conversion failed. Showing raw date range:")
    print(f"From {data['Updated On'].min()} to {data['Updated On'].max()}")

# Data preprocessing
print("\nPreprocessing data...")

# Filter to remove rows where State is missing or where dates are invalid
data = data.dropna(subset=['State', 'Updated On'])

# Define functions to get the latest data with valid values for each state
def get_latest_valid_data(state_group, column_name):
    """Get the most recent row for a state that has a valid (non-zero) value for the specified column."""
    # Sort by date in descending order
    sorted_group = state_group.sort_values('Updated On', ascending=False)
    
    # Find the first row with a valid value
    for idx, row in sorted_group.iterrows():
        if pd.notna(row[column_name]) and row[column_name] > 0:
            return row
    
    # If no valid data found, try to return the latest non-null row
    for idx, row in sorted_group.iterrows():
        if pd.notna(row[column_name]):
            return row
            
    # If still no valid data, return None
    return None

# b. Number of persons state-wise vaccinated for first dose in India
print("\n" + "="*70)
print("B. STATE-WISE FIRST DOSE VACCINATION")
print("="*70)

first_dose_col = 'First Dose Administered'
if first_dose_col in data.columns:
    print(f"Using column '{first_dose_col}' for first dose analysis")
    
    # Get data for each state (exclude India which is the total)
    state_data = data[data['State'] != 'India'].copy()
    
    # Get valid data for each state
    valid_rows = []
    for state, group in state_data.groupby('State'):
        valid_row = get_latest_valid_data(group, first_dose_col)
        if valid_row is not None:
            valid_rows.append(valid_row)
    
    # Convert to DataFrame
    valid_state_data = pd.DataFrame(valid_rows)
    
    # Check if we have valid data
    if not valid_state_data.empty and valid_state_data[first_dose_col].sum() > 0:
        # Sort by first dose count and get top states
        top_states_first = valid_state_data.sort_values(by=first_dose_col, ascending=False)[['State', first_dose_col]].head(10)
        
        print("\nTop 10 States by First Dose Vaccination:")
        for i, (state, doses) in enumerate(zip(top_states_first['State'], top_states_first[first_dose_col]), 1):
            print(f"{i}. {state}: {doses:,.0f}")
        
        # Visualization
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=first_dose_col, y='State', data=top_states_first, palette='viridis')
        
        # Add value labels to bars
        for i, v in enumerate(top_states_first[first_dose_col]):
            ax.text(v + 0.1, i, f"{v:,.0f}", va='center')
            
        plt.title('Top 10 States by First Dose COVID-19 Vaccination', fontsize=14)
        plt.xlabel('Number of First Doses Administered', fontsize=12)
        plt.ylabel('State', fontsize=12)
        plt.tight_layout()
        plt.savefig('state_first_dose.png')
        print("\nFirst dose visualization saved as 'state_first_dose.png'")
    else:
        print("\nNo valid non-zero data found for first dose visualization.")
        print("Taking an alternative approach - using cumulative data...")
        
        # Try looking at the overall data pattern
        pivot_data = data.pivot_table(index='State', values=first_dose_col, aggfunc='max')
        pivot_data = pivot_data[pivot_data.index != 'India'].sort_values(by=first_dose_col, ascending=False).head(10)
        
        if not pivot_data.empty and pivot_data[first_dose_col].sum() > 0:
            print("\nTop 10 States by Maximum First Dose Vaccination (any date):")
            for i, (state, doses) in enumerate(zip(pivot_data.index, pivot_data[first_dose_col]), 1):
                print(f"{i}. {state}: {doses:,.0f}")
            
            # Convert for visualization
            pivot_data = pivot_data.reset_index()
            
            # Visualization
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x=first_dose_col, y='State', data=pivot_data, palette='viridis')
            
            # Add value labels to bars
            for i, v in enumerate(pivot_data[first_dose_col]):
                ax.text(v + 0.1, i, f"{v:,.0f}", va='center')
                
            plt.title('Top 10 States by Maximum First Dose COVID-19 Vaccination', fontsize=14)
            plt.xlabel('Number of First Doses Administered', fontsize=12)
            plt.ylabel('State', fontsize=12)
            plt.tight_layout()
            plt.savefig('state_first_dose.png')
            print("\nFirst dose visualization saved as 'state_first_dose.png'")
        else:
            print("\nStill no valid data found for first dose visualization.")
else:
    print("Column 'First Dose Administered' not found in the dataset!")

# c. Number of persons state-wise vaccinated for second dose in India
print("\n" + "="*70)
print("C. STATE-WISE SECOND DOSE VACCINATION")
print("="*70)

second_dose_col = 'Second Dose Administered'
if second_dose_col in data.columns:
    print(f"Using column '{second_dose_col}' for second dose analysis")
    
    # We'll use the same approach as for first dose
    if 'valid_state_data' in locals() and not valid_state_data.empty:
        # Use the same valid state data from previous section
        top_states_second = valid_state_data.sort_values(by=second_dose_col, ascending=False)[['State', second_dose_col]].head(10)
        
        # Check if we have valid data
        if top_states_second[second_dose_col].sum() > 0:
            print("\nTop 10 States by Second Dose Vaccination:")
            for i, (state, doses) in enumerate(zip(top_states_second['State'], top_states_second[second_dose_col]), 1):
                print(f"{i}. {state}: {doses:,.0f}")
            
            # Visualization
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x=second_dose_col, y='State', data=top_states_second, palette='magma')
            
            # Add value labels to bars
            for i, v in enumerate(top_states_second[second_dose_col]):
                ax.text(v + 0.1, i, f"{v:,.0f}", va='center')
                
            plt.title('Top 10 States by Second Dose COVID-19 Vaccination', fontsize=14)
            plt.xlabel('Number of Second Doses Administered', fontsize=12)
            plt.ylabel('State', fontsize=12)
            plt.tight_layout()
            plt.savefig('state_second_dose.png')
            print("\nSecond dose visualization saved as 'state_second_dose.png'")
        else:
            print("\nNo valid non-zero data found for second dose visualization.")
            print("Taking an alternative approach - using cumulative data...")
            
            # Try looking at the overall data pattern
            pivot_data = data.pivot_table(index='State', values=second_dose_col, aggfunc='max')
            pivot_data = pivot_data[pivot_data.index != 'India'].sort_values(by=second_dose_col, ascending=False).head(10)
            
            if not pivot_data.empty and pivot_data[second_dose_col].sum() > 0:
                print("\nTop 10 States by Maximum Second Dose Vaccination (any date):")
                for i, (state, doses) in enumerate(zip(pivot_data.index, pivot_data[second_dose_col]), 1):
                    print(f"{i}. {state}: {doses:,.0f}")
                
                # Convert for visualization
                pivot_data = pivot_data.reset_index()
                
                # Visualization
                plt.figure(figsize=(12, 8))
                ax = sns.barplot(x=second_dose_col, y='State', data=pivot_data, palette='magma')
                
                # Add value labels to bars
                for i, v in enumerate(pivot_data[second_dose_col]):
                    ax.text(v + 0.1, i, f"{v:,.0f}", va='center')
                    
                plt.title('Top 10 States by Maximum Second Dose COVID-19 Vaccination', fontsize=14)
                plt.xlabel('Number of Second Doses Administered', fontsize=12)
                plt.ylabel('State', fontsize=12)
                plt.tight_layout()
                plt.savefig('state_second_dose.png')
                print("\nSecond dose visualization saved as 'state_second_dose.png'")
            else:
                print("\nStill no valid data found for second dose visualization.")
    else:
        # If we don't have valid state data from first dose section, recompute
        valid_rows = []
        for state, group in state_data.groupby('State'):
            valid_row = get_latest_valid_data(group, second_dose_col)
            if valid_row is not None:
                valid_rows.append(valid_row)
        
        valid_state_data = pd.DataFrame(valid_rows)
        
        if not valid_state_data.empty and valid_state_data[second_dose_col].sum() > 0:
            top_states_second = valid_state_data.sort_values(by=second_dose_col, ascending=False)[['State', second_dose_col]].head(10)
            
            print("\nTop 10 States by Second Dose Vaccination:")
            for i, (state, doses) in enumerate(zip(top_states_second['State'], top_states_second[second_dose_col]), 1):
                print(f"{i}. {state}: {doses:,.0f}")
            
            # Visualization
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x=second_dose_col, y='State', data=top_states_second, palette='magma')
            
            # Add value labels
            for i, v in enumerate(top_states_second[second_dose_col]):
                ax.text(v + 0.1, i, f"{v:,.0f}", va='center')
                
            plt.title('Top 10 States by Second Dose COVID-19 Vaccination', fontsize=14)
            plt.xlabel('Number of Second Doses Administered', fontsize=12)
            plt.ylabel('State', fontsize=12)
            plt.tight_layout()
            plt.savefig('state_second_dose.png')
            print("\nSecond dose visualization saved as 'state_second_dose.png'")
        else:
            print("\nNo valid data found for second dose visualization.")
            print("Taking an alternative approach - using maximum values...")
            
            # Try looking at the overall data pattern
            pivot_data = data.pivot_table(index='State', values=second_dose_col, aggfunc='max')
            pivot_data = pivot_data[pivot_data.index != 'India'].sort_values(by=second_dose_col, ascending=False).head(10)
            
            if not pivot_data.empty and pivot_data[second_dose_col].sum() > 0:
                print("\nTop 10 States by Maximum Second Dose Vaccination (any date):")
                for i, (state, doses) in enumerate(zip(pivot_data.index, pivot_data[second_dose_col]), 1):
                    print(f"{i}. {state}: {doses:,.0f}")
                
                # Convert for visualization
                pivot_data = pivot_data.reset_index()
                
                # Visualization
                plt.figure(figsize=(12, 8))
                ax = sns.barplot(x=second_dose_col, y='State', data=pivot_data, palette='magma')
                
                # Add value labels
                for i, v in enumerate(pivot_data[second_dose_col]):
                    ax.text(v + 0.1, i, f"{v:,.0f}", va='center')
                    
                plt.title('Top 10 States by Maximum Second Dose COVID-19 Vaccination', fontsize=14)
                plt.xlabel('Number of Second Doses Administered', fontsize=12)
                plt.ylabel('State', fontsize=12)
                plt.tight_layout()
                plt.savefig('state_second_dose.png')
                print("\nSecond dose visualization saved as 'state_second_dose.png'")
            else:
                print("\nStill no valid data found for second dose visualization.")
else:
    print("Column 'Second Dose Administered' not found in the dataset!")

# d. Number of Males and Females vaccinated
print("\n" + "="*70)
print("D. GENDER-WISE VACCINATION")
print("="*70)

# Find appropriate columns for gender data
male_ind_col = 'Male(Individuals Vaccinated)'
female_ind_col = 'Female(Individuals Vaccinated)'
male_dose_col = 'Male (Doses Administered)'
female_dose_col = 'Female (Doses Administered)'

# Get data for India (national totals)
india_data = data[data['State'] == 'India'].copy()

# Since India rows are aggregates for each date, we need to find the row with max values
# Function to find the best row for gender data
def get_best_gender_row(df, male_col, female_col):
    # Check if both columns exist
    if male_col in df.columns and female_col in df.columns:
        # Sort by date descending (most recent first)
        sorted_df = df.sort_values('Updated On', ascending=False)
        
        # Try to find a row where both values are present and non-zero
        for idx, row in sorted_df.iterrows():
            if pd.notna(row[male_col]) and pd.notna(row[female_col]) and row[male_col] > 0 and row[female_col] > 0:
                return row, True
        
        # If no row with both values, try to find max values
        max_male = df[male_col].max() if not df[male_col].isna().all() else 0
        max_female = df[female_col].max() if not df[female_col].isna().all() else 0
        
        if max_male > 0 and max_female > 0:
            return pd.Series({male_col: max_male, female_col: max_female}), True
    
    return None, False

# Try to get gender data from individuals vaccinated columns first
row_data, success = get_best_gender_row(india_data, male_ind_col, female_ind_col)

if success:
    print("Using individual vaccination gender data")
    male_count = row_data[male_ind_col]
    female_count = row_data[female_ind_col]
    data_source = "individuals"
else:
    # Try doses administered columns
    row_data, success = get_best_gender_row(india_data, male_dose_col, female_dose_col)
    
    if success:
        print("Using doses administered gender data")
        male_count = row_data[male_dose_col]
        female_count = row_data[female_dose_col]
        data_source = "doses"
    else:
        print("No valid gender data found in national totals. Trying to aggregate from states...")
        
        # Try to aggregate from state data
        valid_states = data[data['State'] != 'India'].copy()
        
        # First try individuals vaccinated
        if male_ind_col in valid_states.columns and female_ind_col in valid_states.columns:
            # For each state, get the maximum values
            state_gender = valid_states.groupby('State').agg({
                male_ind_col: 'max',
                female_ind_col: 'max'
            })
            
            male_count = state_gender[male_ind_col].sum()
            female_count = state_gender[female_ind_col].sum()
            
            if male_count > 0 and female_count > 0:
                print("Aggregated individual vaccination gender data from states")
                data_source = "individuals"
            else:
                # Try doses administered
                state_gender = valid_states.groupby('State').agg({
                    male_dose_col: 'max',
                    female_dose_col: 'max'
                })
                
                male_count = state_gender[male_dose_col].sum()
                female_count = state_gender[female_dose_col].sum()
                
                if male_count > 0 and female_count > 0:
                    print("Aggregated doses administered gender data from states")
                    data_source = "doses"
                else:
                    male_count = 0
                    female_count = 0
                    print("Could not find valid gender data")

# Display gender data if we found it
if male_count > 0 or female_count > 0:
    print(f"\nNumber of Males vaccinated: {male_count:,.0f}")
    print(f"Number of Females vaccinated: {female_count:,.0f}")
    
    # Calculate percentages
    total = male_count + female_count
    male_percent = (male_count / total) * 100
    female_percent = (female_count / total) * 100
    
    print(f"\nMale: {male_percent:.2f}%")
    print(f"Female: {female_percent:.2f}%")
    
    # Create visualization
    plt.figure(figsize=(10, 7))
    plt.pie([male_count, female_count], 
            labels=['Male', 'Female'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#3274A1', '#E1812C'],
            explode=(0, 0.1),
            shadow=True)
    
    if 'data_source' in locals() and data_source == "individuals":
        plt.title('Gender Distribution of COVID-19 Vaccinations in India (Individuals)')
    else:
        plt.title('Gender Distribution of COVID-19 Vaccinations in India (Doses)')
        
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('gender_distribution.png')
    print("\nGender distribution visualization saved as 'gender_distribution.png'")
else:
    print("\nNo valid gender data found for visualization.")

print("\nAnalysis complete!")