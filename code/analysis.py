"""
Senior Selection Sheet Analysis
Object-oriented approach to PDF data analysis using tabula-py
"""

import tabula
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


class PDFReader:
    """Handles PDF reading and table extraction using tabula-py"""
    
    def __init__(self, pdf_path: str):
        """
        Initialize PDF reader with file path
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        self.tables = []
    
    def extract_tables(self, pages: str = 'all', **kwargs) -> List[pd.DataFrame]:
        """
        Extract tables from PDF
        
        Args:
            pages: Pages to extract from (default: 'all')
            **kwargs: Additional arguments for tabula.read_pdf()
        
        Returns:
            List of DataFrames containing extracted tables
        """
        self.tables = tabula.read_pdf(
            str(self.pdf_path),
            pages=pages,
            **kwargs
        )
        return self.tables
    
    def get_table(self, index: int = 0) -> pd.DataFrame:
        """
        Get a specific table by index
        
        Args:
            index: Index of the table to retrieve
        
        Returns:
            DataFrame of the specified table
        """
        if not self.tables:
            raise ValueError("No tables extracted yet. Call extract_tables() first.")
        return self.tables[index]


class DataProcessor:
    """Handles data cleaning and preprocessing"""
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize with a DataFrame
        
        Args:
            dataframe: Input DataFrame to process
        """
        self.df = dataframe.copy()
        self.original_df = dataframe.copy()
    
    def clean_column_names(self) -> 'DataProcessor':
        """Clean and standardize column names"""
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
        return self
    
    def remove_empty_rows(self) -> 'DataProcessor':
        """Remove rows that are completely empty"""
        self.df = self.df.dropna(how='all')
        return self
    
    def remove_empty_columns(self) -> 'DataProcessor':
        """Remove columns that are completely empty"""
        self.df = self.df.dropna(axis=1, how='all')
        return self
    
    def reset_index(self) -> 'DataProcessor':
        """Reset DataFrame index"""
        self.df = self.df.reset_index(drop=True)
        return self
    
    def get_processed_data(self) -> pd.DataFrame:
        """Return the processed DataFrame"""
        return self.df
    
    def restore_original(self) -> 'DataProcessor':
        """Restore to original unprocessed data"""
        self.df = self.original_df.copy()
        return self


class DataAnalyzer:
    """Performs analysis on the processed data"""
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize with a DataFrame
        
        Args:
            dataframe: DataFrame to analyze
        """
        self.df = dataframe
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics for numerical columns"""
        return self.df.describe()
    
    def get_info(self) -> None:
        """Display DataFrame information"""
        return self.df.info()
    
    def get_shape(self) -> tuple:
        """Get shape of DataFrame (rows, columns)"""
        return self.df.shape
    
    def get_column_names(self) -> List[str]:
        """Get list of column names"""
        return self.df.columns.tolist()
    
    def value_counts(self, column: str) -> pd.Series:
        """
        Get value counts for a specific column
        
        Args:
            column: Column name to analyze
        
        Returns:
            Series with value counts
        """
        return self.df[column].value_counts()
    
    def filter_data(self, **conditions) -> pd.DataFrame:
        """
        Filter data based on conditions
        
        Args:
            **conditions: Column-value pairs to filter by
        
        Returns:
            Filtered DataFrame
        """
        filtered_df = self.df.copy()
        for column, value in conditions.items():
            filtered_df = filtered_df[filtered_df[column] == value]
        return filtered_df


class SelectionSheetAnalysis:
    """Main class orchestrating the entire analysis workflow"""
    
    def __init__(self, pdf_path: str):
        """
        Initialize the analysis pipeline
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_reader = PDFReader(pdf_path)
        self.processor = None
        self.analyzer = None
        self.current_data = None
    
    def load_data(self, pages: str = 'all', **kwargs) -> 'SelectionSheetAnalysis':
        """
        Load and extract tables from PDF
        
        Args:
            pages: Pages to extract
            **kwargs: Additional tabula arguments
        
        Returns:
            Self for method chaining
        """
        tables = self.pdf_reader.extract_tables(pages=pages, **kwargs)
        if tables:
            self.current_data = tables[0]  # Use first table by default
            print(f"Loaded {len(tables)} table(s) from PDF")
        return self
    
    def process_data(self) -> 'SelectionSheetAnalysis':
        """
        Process the loaded data
        
        Returns:
            Self for method chaining
        """
        if self.current_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.processor = DataProcessor(self.current_data)
        self.current_data = (self.processor
                            .clean_column_names()
                            .remove_empty_rows()
                            .remove_empty_columns()
                            .reset_index()
                            .get_processed_data())
        
        print("Data processing complete")
        return self
    
    def analyze(self) -> 'SelectionSheetAnalysis':
        """
        Create analyzer instance
        
        Returns:
            Self for method chaining
        """
        if self.current_data is None:
            raise ValueError("No data available. Call load_data() and process_data() first.")
        
        self.analyzer = DataAnalyzer(self.current_data)
        return self
    
    def get_data(self) -> pd.DataFrame:
        """Get current DataFrame"""
        return self.current_data
    
    def display_summary(self) -> None:
        """Display a summary of the current data"""
        if self.analyzer is None:
            raise ValueError("Analyzer not initialized. Call analyze() first.")
        
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"Shape: {self.analyzer.get_shape()}")
        print(f"\nColumns: {self.analyzer.get_column_names()}")
        print("\nFirst few rows:")
        print(self.current_data.head())

    def display_room_stats(self) -> None:
        df = self.get_data()
        #make a bar graph of 'suite_size_(if_applicable)'and count
        suite_counts = df['suite_size_(if_applicable)'].value_counts()
        suite_counts.plot(kind='bar', figsize=(10, 6))
        plt.xlabel('Suite Size')
        plt.ylabel('Count')
        plt.title('Distribution of Suite Sizes')
        plt.tight_layout()
        plt.show()




# Example usage
if __name__ == "__main__":
    # Path to your PDF file
    pdf_path = "/Users/dmakelel/Desktop/MYPRACTICE/senior-selection-sheet-analysis/Data/data.pdf"
    
    # Create analysis instance and run pipeline
    analysis = SelectionSheetAnalysis(pdf_path)
    
    # Load, process, and analyze data
    analysis.load_data().process_data().analyze()
    
    # Display summary
    analysis.display_summary()
    analysis.display_room_stats()
    
    # Get the data for further analysis
    df = analysis.get_data()
    print("\n" + "="*50)
    print("Data is ready for further analysis!")
    print("="*50)
