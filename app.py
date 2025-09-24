# cord19_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CORD19Analyzer:
    def __init__(self):
        self.df = None
        self.cleaned_df = None
        
    def load_data(self, file_path='metadata.csv'):
        """Load the CORD-19 metadata"""
        try:
            self.df = pd.read_csv(file_path)
            print("✅ Data loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"❌ File {file_path} not found!")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def basic_exploration(self):
        """Perform basic data exploration"""
        if self.df is None:
            print("❌ No data loaded!")
            return
            
        print("\n" + "="*50)
        print("BASIC DATA EXPLORATION")
        print("="*50)
        
        # 1. First few rows
        print("\n📊 First 5 rows:")
        print(self.df.head())
        
        # 2. DataFrame dimensions
        print(f"\n📐 Dataset Dimensions: {self.df.shape}")
        print(f"   - Rows: {self.df.shape[0]}")
        print(f"   - Columns: {self.df.shape[1]}")
        
        # 3. Data types
        print("\n🔍 Data Types:")
        print(self.df.dtypes)
        
        # 4. Missing values
        print("\n❓ Missing Values Summary:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        })
        print(missing_df.sort_values('Missing Count', ascending=False).head(10))
        
        # 5. Basic statistics for numerical columns
        print("\n📈 Basic Statistics:")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(self.df[numerical_cols].describe())
        else:
            print("No numerical columns found.")

    def clean_data(self):
        """Clean and prepare the data for analysis"""
        if self.df is None:
            print("❌ No data loaded!")
            return
            
        print("\n" + "="*50)
        print("DATA CLEANING AND PREPARATION")
        print("="*50)
        
        # Create a copy for cleaning
        self.cleaned_df = self.df.copy()
        
        # 1. Handle missing values in important columns
        important_cols = ['title', 'abstract', 'journal', 'publish_time']
        
        print("\n🧹 Handling missing values:")
        for col in important_cols:
            if col in self.cleaned_df.columns:
                missing_count = self.cleaned_df[col].isnull().sum()
                print(f"   - {col}: {missing_count} missing values ({missing_count/len(self.cleaned_df)*100:.1f}%)")
        
        # Keep only rows with titles (essential for analysis)
        initial_rows = len(self.cleaned_df)
        self.cleaned_df = self.cleaned_df.dropna(subset=['title'])
        removed_rows = initial_rows - len(self.cleaned_df)
        print(f"   - Removed {removed_rows} rows without titles")
        
        # 2. Convert date columns
        if 'publish_time' in self.cleaned_df.columns:
            print("\n📅 Processing dates:")
            # Convert to datetime, handling errors
            self.cleaned_df['publish_time'] = pd.to_datetime(
                self.cleaned_df['publish_time'], errors='coerce'
            )
            
            # Extract year for time-based analysis
            self.cleaned_df['publication_year'] = self.cleaned_df['publish_time'].dt.year
            
            # Check year distribution
            year_counts = self.cleaned_df['publication_year'].value_counts().sort_index()
            print("   - Publication years found:", year_counts.index.tolist())
        
        # 3. Create new columns for analysis
        print("\n✨ Creating new features:")
        
        # Abstract word count
        if 'abstract' in self.cleaned_df.columns:
            self.cleaned_df['abstract_word_count'] = self.cleaned_df['abstract'].apply(
                lambda x: len(str(x).split()) if pd.notnull(x) else 0
            )
            print("   - Created abstract_word_count")
        
        # Title word count
        if 'title' in self.cleaned_df.columns:
            self.cleaned_df['title_word_count'] = self.cleaned_df['title'].apply(
                lambda x: len(str(x).split()) if pd.notnull(x) else 0
            )
            print("   - Created title_word_count")
        
        print(f"✅ Cleaning complete! Final shape: {self.cleaned_df.shape}")
        return self.cleaned_df
    
    def analyze_data(self):
        """Perform data analysis and create visualizations"""
        if self.cleaned_df is None:
            print("❌ No cleaned data available!")
            return
            
        print("\n" + "="*50)
        print("DATA ANALYSIS AND VISUALIZATION")
        print("="*50)
        
        # Create visualizations
        self._create_publications_over_time()
        self._create_top_journals_chart()
        self._create_title_word_cloud()
        self._create_source_distribution()
        
    def _create_publications_over_time(self):
        """Plot number of publications over time"""
        if 'publication_year' not in self.cleaned_df.columns:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Count publications by year
        year_counts = self.cleaned_df['publication_year'].value_counts().sort_index()
        
        # Filter out invalid years
        year_counts = year_counts[year_counts.index >= 2019]
        
        plt.bar(year_counts.index, year_counts.values, color='skyblue', alpha=0.8)
        plt.title('📈 COVID-19 Publications Over Time', fontweight='bold', pad=20)
        plt.xlabel('Publication Year')
        plt.ylabel('Number of Publications')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(year_counts.values):
            plt.text(year_counts.index[i], v + 10, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('publications_over_time.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Publications by year: {dict(year_counts)}")
    
    def _create_top_journals_chart(self):
        """Create bar chart of top publishing journals"""
        if 'journal' not in self.cleaned_df.columns:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Get top 15 journals
        top_journals = self.cleaned_df['journal'].value_counts().head(15)
        
        # Plot horizontal bar chart for better readability
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_journals)))
        bars = plt.barh(range(len(top_journals)), top_journals.values, color=colors, alpha=0.8)
        
        plt.title('🏥 Top Journals Publishing COVID-19 Research', fontweight='bold', pad=20)
        plt.xlabel('Number of Publications')
        plt.yticks(range(len(top_journals)), top_journals.index)
        plt.gca().invert_yaxis()  # Highest count at top
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(top_journals.values):
            plt.text(v + 10, i, str(v), va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('top_journals.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📋 Top 5 journals:")
        for journal, count in top_journals.head().items():
            print(f"   - {journal}: {count} publications")
    
    def _create_title_word_cloud(self):
        """Generate word cloud from paper titles"""
        if 'title' not in self.cleaned_df.columns:
            return
            
        # Combine all titles
        all_titles = ' '.join(self.cleaned_df['title'].dropna().astype(str))
        
        # Remove common words and prepare text
        stop_words = ['the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'by', 'an', 'from']
        
        plt.figure(figsize=(15, 8))
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            stopwords=stop_words,
            max_words=100,
            colormap='viridis'
        ).generate(all_titles)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('☁️ Most Frequent Words in Paper Titles', fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('title_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Show most frequent words
        from collections import Counter
        words = all_titles.lower().split()
        words = [word for word in words if word not in stop_words and len(word) > 3]
        word_freq = Counter(words).most_common(10)
        
        print("\n🔤 Top 10 words in titles:")
        for word, count in word_freq:
            print(f"   - {word}: {count} occurrences")
    
    def _create_source_distribution(self):
        """Plot distribution of paper counts by source"""
        if 'source_x' not in self.cleaned_df.columns:
            # Try to find source column
            source_cols = [col for col in self.cleaned_df.columns if 'source' in col.lower()]
            if not source_cols:
                return
            source_col = source_cols[0]
        else:
            source_col = 'source_x'
        
        plt.figure(figsize=(12, 6))
        
        source_counts = self.cleaned_df[source_col].value_counts().head(10)
        
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(source_counts))))
        plt.title('📚 Distribution of Papers by Source', fontweight='bold', pad=20)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('source_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()


    def create_streamlit_app(self):
        """Create and run the Streamlit application"""
        
        st.set_page_config(
            page_title="CORD-19 Data Explorer",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title and description
        st.title("🔬 CORD-19 COVID-19 Research Explorer")
        st.markdown("""
        This interactive dashboard explores the CORD-19 dataset containing metadata 
        about COVID-19 research papers. Use the filters below to explore the data.
        """)
        
        # Sidebar for filters
        st.sidebar.header("🔍 Filters and Controls")
        
        if self.cleaned_df is None:
            st.warning("Please load and clean the data first!")
            return
        
        # Year range slider
        if 'publication_year' in self.cleaned_df.columns:
            min_year = int(self.cleaned_df['publication_year'].min())
            max_year = int(self.cleaned_df['publication_year'].max())
            
            year_range = st.sidebar.slider(
                "Select Publication Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        else:
            year_range = (2019, 2022)
        
        # Journal filter
        if 'journal' in self.cleaned_df.columns:
            journals = ['All'] + self.cleaned_df['journal'].dropna().unique().tolist()
            selected_journal = st.sidebar.selectbox("Select Journal", journals)
        else:
            selected_journal = "All"
        
        # Filter data based on selections
        filtered_df = self.cleaned_df.copy()
        
        if 'publication_year' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['publication_year'] >= year_range[0]) & 
                (filtered_df['publication_year'] <= year_range[1])
            ]
        
        if selected_journal != "All":
            filtered_df = filtered_df[filtered_df['journal'] == selected_journal]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Papers", len(filtered_df))
        
        with col2:
            if 'publication_year' in filtered_df.columns:
                st.metric("Years Covered", f"{year_range[0]} - {year_range[1]}")
        
        with col3:
            if 'journal' in filtered_df.columns:
                unique_journals = filtered_df['journal'].nunique()
                st.metric("Unique Journals", unique_journals)
        
        with col4:
            if 'abstract_word_count' in filtered_df.columns:
                avg_words = int(filtered_df['abstract_word_count'].mean())
                st.metric("Avg Abstract Words", avg_words)
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview", "🏥 Journals", "☁️ Word Analysis", "📊 Data Sample"
        ])
        
        with tab1:
            st.subheader("Publication Trends")
            
            if 'publication_year' in filtered_df.columns:
                year_counts = filtered_df['publication_year'].value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(year_counts.index, year_counts.values, color='lightblue', alpha=0.8)
                ax.set_title('Publications by Year')
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of Publications')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
            else:
                st.info("Publication year data not available for filtering")
        
        with tab2:
            st.subheader("Top Publishing Journals")
            
            if 'journal' in filtered_df.columns:
                top_journals = filtered_df['journal'].value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = plt.cm.Paired(np.linspace(0, 1, len(top_journals)))
                ax.barh(range(len(top_journals)), top_journals.values, color=colors)
                ax.set_yticks(range(len(top_journals)))
                ax.set_yticklabels(top_journals.index)
                ax.invert_yaxis()
                ax.set_xlabel('Number of Publications')
                ax.set_title('Top 10 Journals')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
        
        with tab3:
            st.subheader("Word Frequency Analysis")
            
            if 'title' in filtered_df.columns:
                # Word cloud
                st.write("### Word Cloud of Paper Titles")
                all_titles = ' '.join(filtered_df['title'].dropna().astype(str))
                
                if all_titles.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("No title data available for selected filters")
        
        with tab4:
            st.subheader("Data Sample")
            
            # Show sample data
            st.write(f"Showing 10 random papers from the filtered set ({len(filtered_df)} total):")
            
            sample_cols = ['title', 'journal', 'publication_year', 'abstract_word_count']
            available_cols = [col for col in sample_cols if col in filtered_df.columns]
            
            if available_cols:
                st.dataframe(
                    filtered_df[available_cols].sample(min(10, len(filtered_df))).reset_index(drop=True),
                    height=400
                )
            
            # Data summary
            st.write("### Data Summary")
            st.write(filtered_df.describe(include='all').T)


def main():
    """Main function to run the complete analysis"""
    print("🚀 Starting CORD-19 Data Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = CORD19Analyzer()
    
    try:
        # Part 1: Load data
        print("\n📥 PART 1: Loading Data...")
        if not analyzer.load_data('metadata.csv'):
            # Try alternative path or create sample data
            print("⚠️  Could not load metadata.csv, using sample data...")
            # You would need to implement sample data creation here
            return
        
        # Basic exploration
        analyzer.basic_exploration()
        
        # Part 2: Clean data
        print("\n🧹 PART 2: Cleaning Data...")
        analyzer.clean_data()
        
        # Part 3: Analysis and visualization
        print("\n📊 PART 3: Analyzing Data...")
        analyzer.analyze_data()
        
        # Part 4: Streamlit app
        print("\n🌐 PART 4: Creating Streamlit App...")
        
        # Note: Streamlit apps are typically run separately
        # For demonstration, we'll show how to structure it
        st_app_code = """
        # Save this as app.py and run with: streamlit run app.py
        
        import streamlit as st
        from cord19_analysis import CORD19Analyzer
        
        analyzer = CORD19Analyzer()
        analyzer.load_data('metadata.csv')
        analyzer.clean_data()
        analyzer.create_streamlit_app()
        """
        
        print("💡 To run the Streamlit app:")
        print("1. Save the Streamlit code in a separate file (app.py)")
        print("2. Run: streamlit run app.py")
        
        # Part 5: Documentation
        print("\n📝 PART 5: Documentation and Reflection")
        print("\n🔍 Key Findings:")
        print("- The dataset contains metadata for COVID-19 research papers")
        print("- Analysis shows publication trends over time")
        print("- Top journals publishing COVID-19 research can be identified")
        print("- Word frequency analysis reveals common research themes")
        
        print("\n💡 Challenges and Learning:")
        print("- Handling missing values in real-world datasets")
        print("- Working with datetime data for time-series analysis")
        print("- Creating interactive visualizations with Streamlit")
        print("- Processing text data for word frequency analysis")
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()