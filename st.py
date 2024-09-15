import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose


# Set page configuration
st.set_page_config(layout="wide", page_title="Emotion and Transcript Data Dashboard")

# Sidebar
st.sidebar.title("Navigation")
data_type = st.sidebar.radio("Select Data Type", [
    "final",
    "Emotion Data", 
    "Transcript Data", 
    "Combined Emotion Analysis", 
    "Combined Gaze Data", 
    "Emotion with Gaze and Transcript Analysis"
    
])

# Helper functions
@st.cache_data
def load_all_emotion_data():
    all_data = []
    for option in range(1, 11):
        df = pd.read_csv(f"emotion_data/{option}/emotion.csv")
        df['option'] = option
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def extract_top_emotions(df):
    top_emotions = df['dominant_emotion'].value_counts().nlargest(4).index.tolist()
    return pd.Series({
        'movie_id': df['movie_id'].iloc[0],
        'top_emotion_1': top_emotions[0],
        'top_emotion_2': top_emotions[1] if len(top_emotions) > 1 else None
    })

@st.cache_data
def load_all_transcript_data():
    all_transcript_data = []
    for option in range(1, 11):
        df = pd.read_csv(f"transcript_data/{option}.csv")
        df['option'] = option
        all_transcript_data.append(df)
    return pd.concat(all_transcript_data, ignore_index=True)

@st.cache_data
def load_all_gaze_data():
    all_gaze_data = []
    for option in range(1, 11):
        df = pd.read_csv(f"emotion_data/{option}/gaze.csv")
        df['option'] = option
        all_gaze_data.append(df)
    return pd.concat(all_gaze_data, ignore_index=True)

@st.cache_data
def load_emotion_data(option):
    emotion_df = pd.read_csv(f"emotion_data/{option}/emotion.csv")
    gaze_df = pd.read_csv(f"emotion_data/{option}/gaze.csv")
    return emotion_df, gaze_df

@st.cache_data
def load_transcript_data(option):
    transcript_df = pd.read_csv(f"transcript_data/{option}.csv")
    return transcript_df


def emotion_data_analysis(option):
    emotion_df, gaze_df = load_emotion_data(option)
    
    st.title(f"Emotion Data Analysis - Student {option}")
    st.subheader("Emotion Data Sample")
    st.write(emotion_df.head())

    st.subheader("Basic Statistics")
    st.write(emotion_df.describe())

    st.subheader("Emotion Distribution")
    fig = px.histogram(emotion_df, x='dominant_emotion', title="Distribution of Dominant Emotions")
    st.plotly_chart(fig)

    st.subheader("Emotion Intensity Heatmap")
    emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    fig = px.imshow(emotion_df[emotion_columns].transpose(), 
                    labels=dict(x="Frame", y="Emotion", color="Intensity"),
                    title="Emotion Intensity Heatmap")
    st.plotly_chart(fig)

    st.subheader("Emotion Transition Matrix")
    emotion_df['prev_emotion'] = emotion_df['dominant_emotion'].shift(1)
    transition_matrix = pd.crosstab(emotion_df['prev_emotion'], emotion_df['dominant_emotion'], normalize='index')
    fig = px.imshow(transition_matrix, labels=dict(x="To", y="From", color="Transition Probability"),
                    title="Emotion Transition Matrix")
    st.plotly_chart(fig)

    st.subheader("Average Emotion Intensity Radar Chart")
    avg_emotions = emotion_df[emotion_columns].mean().values.tolist()
    fig = go.Figure(data=go.Scatterpolar(
        r=avg_emotions + [avg_emotions[0]],
        theta=emotion_columns + [emotion_columns[0]],
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="Average Emotion Intensity"
    )
    st.plotly_chart(fig)

    st.subheader("Emotion Intensities Over Time")
    emotion_intensities = emotion_df.melt(id_vars=['movie_id', 'image_seq'], 
                                          value_vars=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
                                          var_name='emotion', value_name='intensity')
    fig = px.line(emotion_intensities, x='image_seq', y='intensity', color='emotion', title="Emotion Intensities Over Time")
    st.plotly_chart(fig)

    st.subheader("Emotion Correlation Heatmap")
    corr = emotion_df[['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Emotion Correlation Heatmap")
    st.plotly_chart(fig)

    st.subheader("Boxplot of Emotion Intensities")
    fig = px.box(emotion_intensities, x='emotion', y='intensity', title="Distribution of Emotion Intensities")
    st.plotly_chart(fig)

    st.subheader("3D Scatter Plot of Top 3 Emotions")
    top_3_emotions = emotion_df[emotion_columns].mean().nlargest(3).index.tolist()
    fig = px.scatter_3d(emotion_df, x=top_3_emotions[0], y=top_3_emotions[1], z=top_3_emotions[2],
                        color='dominant_emotion', title=f"3D Scatter of {', '.join(top_3_emotions)}")
    st.plotly_chart(fig)


def gaze_data_analysis(option):
    emotion_df, gaze_df = load_emotion_data(option)
    
    st.header("Gaze Data Analysis")

    st.header("Structure of Gaze Data")
    st.write(gaze_df.head())
    
    st.subheader("Basic Statistics")
    st.write(gaze_df.describe())

    st.subheader("Gaze and Blink Over Time")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Gaze Over Time", "Blink Over Time"))
    fig.add_trace(go.Scatter(x=gaze_df['image_seq'], y=gaze_df['gaze'], mode='lines', name='Gaze'), row=1, col=1)
    fig.add_trace(go.Scatter(x=gaze_df['image_seq'], y=gaze_df['blink'], mode='lines', name='Blink'), row=2, col=1)
    fig.update_layout(height=600, title_text="Gaze and Blink Over Time")
    st.plotly_chart(fig)

    st.subheader("Eye Offset Distribution")
    fig = px.histogram(gaze_df, x='eye_offset', title="Eye Offset Distribution")
    st.plotly_chart(fig)

    st.subheader("Gaze vs Blink Scatter Plot")
    fig = px.scatter(gaze_df, x='gaze', y='blink', color='eye_offset', title="Gaze vs Blink (colored by Eye Offset)")
    st.plotly_chart(fig)

    st.subheader("Gaze Time Series Decomposition")
    gaze_series = pd.Series(gaze_df['gaze'].values, index=pd.date_range(start='2021-01-01', periods=len(gaze_df), freq='D'))
    decomposition = seasonal_decompose(gaze_series, model='additive', period=30)
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
    fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
    fig.update_layout(height=900, title_text="Gaze Time Series Decomposition")
    st.plotly_chart(fig)


def transcript_data_analysis(option):
    transcript_df = load_transcript_data(option)

    st.title(f"Transcript Data Analysis - Option {option}")
    
    st.subheader("Transcript Data Sample")
    st.write(transcript_df.head())

    st.subheader("Basic Statistics")
    st.write(transcript_df.describe())

    st.subheader("Speech Duration Analysis")
    transcript_df['duration'] = transcript_df['end'] - transcript_df['start']
    fig = px.histogram(transcript_df, x='duration', title="Distribution of Speech Durations")
    st.plotly_chart(fig)


    # Sentiment analysis
    st.subheader("Sentiment Analysis")
    sentiment_cols = ['positive', 'negative', 'neutral']
    fig = px.bar(transcript_df, x=transcript_df.index, y=sentiment_cols, 
                 title="Sentiment Scores for Each Speech Segment")
    st.plotly_chart(fig)

    st.subheader("Speech Speed and Confidence Over Time")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=transcript_df['start'], y=transcript_df['speech_speed'], 
                            mode='lines', name='Speech Speed'), secondary_y=False)
    fig.add_trace(go.Scatter(x=transcript_df['start'], y=transcript_df['confident'], 
                            mode='lines', name='Confidence', line=dict(dash='dash')), secondary_y=True)
    fig.update_layout(title="Speech Speed and Confidence Over Time")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Speech Speed", secondary_y=False)
    fig.update_yaxes(title_text="Confidence", secondary_y=True)
    st.plotly_chart(fig)

    st.subheader("Speech Characteristics Radar Chart")
    speech_chars = ['confident', 'hesitant', 'concise', 'enthusiastic']
    avg_chars = transcript_df[speech_chars].mean().values.tolist()
    fig = go.Figure(data=go.Scatterpolar(
        r=avg_chars + [avg_chars[0]],
        theta=speech_chars + [speech_chars[0]],
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        title="Average Speech Characteristics"
    )
    st.plotly_chart(fig)
    

    # Correlation Heatmap for Speech Characteristics
    st.subheader("Correlation Heatmap for Speech Characteristics")
    corr = transcript_df[speech_chars].corr()
    fig = px.imshow(corr, labels=dict(x="Characteristics", y="Characteristics", color="Correlation"),
                    title="Correlation Heatmap of Speech Characteristics")
    st.plotly_chart(fig)


    # Speech speed over time
    st.subheader("Speech Speed Over Time")
    fig = px.line(transcript_df, x='start', y='speech_speed', 
                  title="Speech Speed Over Time")
    st.plotly_chart(fig)
    
    # Word count vs Speech speed scatter plot
    st.subheader("Word Count vs Speech Speed")
    transcript_df['word_count'] = transcript_df['text'].str.split().str.len()
    fig = px.scatter(transcript_df, x='word_count', y='speech_speed', 
                     color='confident', size='enthusiastic',
                     title="Word Count vs Speech Speed (colored by confidence, size by enthusiasm)")
    st.plotly_chart(fig)


def combined_emotion_analysis():
        st.title("Combined Emotion Analysis")
        all_emotion_data = load_all_emotion_data()
    
    # Extract top emotions for each movie
        top_emotions_df = all_emotion_data.groupby('option').apply(extract_top_emotions).reset_index()
        st.subheader("Top Emotions for Each Movie")
        st.write(top_emotions_df)

        # Visualize top emotions distribution
        st.subheader("Distribution of Top Emotions")
        top_emotions = pd.concat([top_emotions_df['top_emotion_1'], top_emotions_df['top_emotion_2']]).value_counts()
        fig = px.bar(x=top_emotions.index, y=top_emotions.values, labels={'x': 'Emotion', 'y': 'Count'},
                     title="Distribution of Top Emotions Across All Movies")
        st.plotly_chart(fig)

        # Average emotion intensities across all movies
        st.subheader("Average Emotion Intensities Across All Movies")
        avg_emotions = all_emotion_data[['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']].mean()
        fig = px.bar(x=avg_emotions.index, y=avg_emotions.values, labels={'x': 'Emotion', 'y': 'Average Intensity'},
                     title="Average Emotion Intensities Across All Movies")
        st.plotly_chart(fig)

        # Emotion intensity heatmap across all movies
        st.subheader("Emotion Intensity Heatmap Across All Movies")
        emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        heatmap_data = all_emotion_data.groupby('option')[emotion_columns].mean()
        fig = px.imshow(heatmap_data.transpose(), 
                        labels=dict(x="Movie Option", y="Emotion", color="Average Intensity"),
                        title="Emotion Intensity Heatmap Across All Movies")
        st.plotly_chart(fig)

        # Correlation between emotions across all movies
        st.subheader("Correlation Between Emotions Across All Movies")
        corr_matrix = all_emotion_data[emotion_columns].corr()
        fig = px.imshow(corr_matrix, 
                        labels=dict(x="Emotion", y="Emotion", color="Correlation"),
                        title="Correlation Between Emotions Across All Movies")
        st.plotly_chart(fig)

        # Box plot of emotion intensities across all movies
        st.subheader("Distribution of Emotion Intensities Across All Movies")
        emotion_data_melted = all_emotion_data.melt(value_vars=emotion_columns, var_name='Emotion', value_name='Intensity')
        fig = px.box(emotion_data_melted, x='Emotion', y='Intensity', 
                     title="Distribution of Emotion Intensities Across All Movies")
        st.plotly_chart(fig)


        st.subheader("Emotion Intensity Over Time (All Movies)")
        emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_data_melted = all_emotion_data.melt(id_vars=['option', 'image_seq'], value_vars=emotion_columns, var_name='Emotion', value_name='Intensity')
        fig = px.line(emotion_data_melted, x='image_seq', y='Intensity', color='Emotion', facet_col='option', facet_col_wrap=2,
                    title="Emotion Intensity Over Time for All Movies")
        st.plotly_chart(fig)

        st.subheader("Dominant Emotion Transitions")
        all_emotion_data['prev_emotion'] = all_emotion_data.groupby('option')['dominant_emotion'].shift(1)
        transition_matrix = pd.crosstab(all_emotion_data['prev_emotion'], all_emotion_data['dominant_emotion'], normalize='index')
        fig = px.imshow(transition_matrix, labels=dict(x="To", y="From", color="Transition Probability"),
                        title="Emotion Transition Matrix (All Movies)")
        st.plotly_chart(fig)

        st.subheader("Emotion Variability Across Movies")
        emotion_std = all_emotion_data.groupby('option')[emotion_columns].std()
        fig = px.bar(emotion_std.reset_index(), x='option', y=emotion_columns, 
                    title="Emotion Variability (Standard Deviation) Across Movies")
        st.plotly_chart(fig)


def combined_gaze_analysis():
        st.title("Combined Gaze Data Analysis")

        # Load all gaze data
        all_gaze_data = load_all_gaze_data()

        # Basic Gaze Data Stats
        st.subheader("Gaze Data Statistics")
        st.write(all_gaze_data.describe())

        # Gaze and Blink Distribution
        st.subheader("Gaze and Blink Distribution")
        fig = px.histogram(all_gaze_data, x='gaze', color='blink',
                           title="Distribution of Gaze with Blink Information")
        st.plotly_chart(fig)

        # Gaze vs. Eye Offset Scatter Plot
        st.subheader("Gaze vs. Eye Offset")
        fig = px.scatter(all_gaze_data, x='gaze', y='eye_offset', color='blink',
                         title="Gaze vs. Eye Offset (colored by Blink)")
        st.plotly_chart(fig)

        # Time series of gaze and eye offset
        st.subheader("Gaze and Eye Offset Over Time")
        fig = px.line(all_gaze_data, x='image_seq', y=['gaze', 'eye_offset'], 
                      title="Gaze and Eye Offset Over Time")
        st.plotly_chart(fig)

        st.subheader("Gaze Patterns Across Movies")
        fig = px.box(all_gaze_data, x='option', y='gaze', title="Gaze Distribution Across Movies")
        st.plotly_chart(fig)

        st.subheader("Blink Rate Over Time (All Movies)")
        fig = px.line(all_gaze_data, x='image_seq', y='blink', color='option', 
                    title="Blink Rate Over Time for All Movies")
        st.plotly_chart(fig)

        st.subheader("Eye Offset vs Gaze (All Movies)")
        fig = px.scatter(all_gaze_data, x='gaze', y='eye_offset', color='option', 
                        title="Eye Offset vs Gaze for All Movies")
        st.plotly_chart(fig)

        st.subheader("Gaze Heatmap Across Movies")
        gaze_pivot = all_gaze_data.pivot(index='image_seq', columns='option', values='gaze')
        fig = px.imshow(gaze_pivot, title="Gaze Heatmap Across Movies")
        st.plotly_chart(fig)



# Load data
def show_final_analysis():
    st.title("Advanced EDA of Video Text Dataset")
    
    df = pd.read_csv("final_df.csv")

    # Display first few rows of the dataset
    st.write("### Data Preview:")
    st.write(df.head())

    
     # General statistics
    st.write("### Data Info:")
    
    # Manually extract data from df.info() to construct a DataFrame
    info_data = {
        'Column': df.columns,
        'Non-Null Count': [df[col].notnull().sum() for col in df.columns],
        'Dtype': [df[col].dtype for col in df.columns]
    }
    
    df_info = pd.DataFrame(info_data)
    
    # Display the DataFrame
    st.write(df_info)

    st.write("### Data Summary:")
    st.write(df.describe())

    # Correlation matrix (excluding non-numeric columns)
    st.write("### Correlation Matrix:")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    corr = numeric_df.corr()  # Compute correlation only for numeric columns
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
    st.plotly_chart(fig)

    def distribution_plots(df):
        st.write("### Distribution of Emotional and Communication Attributes:")
        
        # List of columns to visualize
        columns = ['avg_positive', 'avg_negative', 'avg_neutral', 'avg_confident', 
                'avg_hesitant', 'avg_concise', 'avg_enthusiastic', 'avg_speech_speed']
        
        for col in columns:
            st.write(f"#### {col} Distribution:")
            fig = px.line(df,x= range(1,11), y=col, title=f"Distribution of {col}")
            st.plotly_chart(fig)

    # Call the function
    distribution_plots(df)
    def analyze_confidence(df):
        st.write("### High-Confidence vs Low-Confidence Groups:")
        
        # Define threshold for high confidence
        high_conf_threshold = df['avg_confident'].median()
        
        # Create high and low confidence groups
        high_confidence = df[df['avg_confident'] >= high_conf_threshold]
        low_confidence = df[df['avg_confident'] < high_conf_threshold]
        
        st.write("#### High-Confidence Group:")
        st.write(high_confidence[['avg_positive', 'avg_negative', 'avg_neutral', 'avg_confident', 'avg_hesitant', 'avg_speech_speed']])
        
        st.write("#### Low-Confidence Group:")
        st.write(low_confidence[['avg_positive', 'avg_negative', 'avg_neutral', 'avg_confident', 'avg_hesitant', 'avg_speech_speed']])

    # Call the function
    analyze_confidence(df)

    def enthusiastic_neutral(df):
        st.write("### Enthusiastic Students with Neutral Text Score:")
        
        # Define threshold for high enthusiasm
        enth_threshold = df['avg_enthusiastic'].median()
        
        # Enthusiastic students with a neutral dominant emotion
        enth_neutral = df[(df['avg_enthusiastic'] >= enth_threshold) & (df['dominant_emotion_top1'] == 'neutral')]
        
        st.write(f"Number of enthusiastic students with neutral text score: {len(enth_neutral)}")
        
        # Show these students
        st.write(enth_neutral[['avg_positive', 'avg_confident', 'avg_enthusiastic', 'avg_speech_speed', 'dominant_emotion_top1']])

    # Call the function
    enthusiastic_neutral(df)


    def analyze_positive_confident_enthusiastic(df):
        st.write("### Positive, Confident, Enthusiastic (with Speech Speed and Hesitance):")
        
        # Define thresholds
        pos_threshold = df['avg_positive'].median()
        conf_threshold = df['avg_confident'].median()
        enth_threshold = df['avg_enthusiastic'].median()
        speech_speed_threshold = df['avg_speech_speed'].median()
        hesitant_threshold = df['avg_hesitant'].median()
        
        # Positive, confident, and enthusiastic with normal speech speed and low hesitation
        pos_conf_enth_norm_speed = df[(df['avg_positive'] >= pos_threshold) & 
                                    (df['avg_confident'] >= conf_threshold) &
                                    (df['avg_enthusiastic'] >= enth_threshold) &
                                    (df['avg_speech_speed'] <= speech_speed_threshold) &
                                    (df['avg_hesitant'] <= hesitant_threshold)]
        
        st.write(f"Number of students who are positive, confident, enthusiastic, with normal speech speed and low hesitance: {len(pos_conf_enth_norm_speed)}")
        
        # Show these students
        st.write(pos_conf_enth_norm_speed[['avg_positive', 'avg_confident', 'avg_enthusiastic', 'avg_speech_speed', 'avg_hesitant']])

    # Call the function
    analyze_positive_confident_enthusiastic(df)

    def analyze_positive_confident_enthusiastic2(df):
        st.write("### Positive, Confident, and Enthusiastic Students:")
        
        # Define thresholds
        pos_threshold = df['avg_positive'].median()
        conf_threshold = df['avg_confident'].median()
        enth_threshold = df['avg_enthusiastic'].median()
        concise_threshold = df['avg_concise'].median()
        
        # Positive and confident
        pos_conf = df[(df['avg_positive'] >= pos_threshold) & (df['avg_confident'] >= conf_threshold)]
        st.write(f"Number of students who are positive and confident: {len(pos_conf)}")
        
        # Positive and enthusiastic
        pos_enth = df[(df['avg_positive'] >= pos_threshold) & (df['avg_enthusiastic'] >= enth_threshold)]
        st.write(f"Number of students who are positive and enthusiastic: {len(pos_enth)}")
        
        # Positive but not concise
        pos_not_concise = df[(df['avg_positive'] >= pos_threshold) & (df['avg_concise'] < concise_threshold)]
        st.write(f"Number of students who are positive but not concise: {len(pos_not_concise)}")
        
        # Positive, confident but not concise
        pos_conf_not_concise = df[(df['avg_positive'] >= pos_threshold) & 
                                (df['avg_confident'] >= conf_threshold) & 
                                (df['avg_concise'] < concise_threshold)]
        st.write(f"Number of students who are positive and confident but not concise: {len(pos_conf_not_concise)}")
        
        # Confident but not positive
        conf_not_pos = df[(df['avg_confident'] >= conf_threshold) & (df['avg_positive'] < pos_threshold)]
        st.write(f"Number of students who are confident but not positive: {len(conf_not_pos)}")

    # Call the function
    analyze_positive_confident_enthusiastic2(df)
    

    def analyze_emotions_and_attributes_interactive(df):
        st.write("### Interactive Emotional and Attribute-Based Analysis")
        
        # Define thresholds
        pos_threshold = df['avg_positive'].median()
        conf_threshold = df['avg_confident'].median()
        enth_threshold = df['avg_enthusiastic'].median()
        concise_threshold = df['avg_concise'].median()

        # Dropdown to filter by dominant emotion
        emotion = st.selectbox("Select Dominant Emotion to Analyze:", ['All'] + df['dominant_emotion_top1'].unique().tolist())

        # Filter the dataframe based on the selected emotion
        if emotion != 'All':
            filtered_df = df[df['dominant_emotion_top1'] == emotion]
        else:
            filtered_df = df
        
        st.write(f"Displaying data for emotion: {emotion}")

        # Positive and confident
        pos_conf = filtered_df[(filtered_df['avg_positive'] >= pos_threshold) & (filtered_df['avg_confident'] >= conf_threshold)]
        st.write(f"Number of students who are positive and confident: {len(pos_conf)}")

        # Positive and enthusiastic
        pos_enth = filtered_df[(filtered_df['avg_positive'] >= pos_threshold) & (filtered_df['avg_enthusiastic'] >= enth_threshold)]
        st.write(f"Number of students who are positive and enthusiastic: {len(pos_enth)}")

        # Positive but not concise
        pos_not_concise = filtered_df[(filtered_df['avg_positive'] >= pos_threshold) & (filtered_df['avg_concise'] < concise_threshold)]
        st.write(f"Number of students who are positive but not concise: {len(pos_not_concise)}")

        # Confident but not positive
        conf_not_pos = filtered_df[(filtered_df['avg_confident'] >= conf_threshold) & (filtered_df['avg_positive'] < pos_threshold)]
        st.write(f"Number of students who are confident but not positive: {len(conf_not_pos)}")

        # Positive, confident but not concise
        pos_conf_not_concise = filtered_df[(filtered_df['avg_positive'] >= pos_threshold) & 
                                            (filtered_df['avg_confident'] >= conf_threshold) & 
                                            (filtered_df['avg_concise'] < concise_threshold)]
        st.write(f"Number of students who are positive and confident but not concise: {len(pos_conf_not_concise)}")
        
        # Plot positive vs. confident
        fig1 = px.scatter(filtered_df, x='avg_positive', y='avg_confident', 
                        color='dominant_emotion_top1', 
                        title='Positive vs. Confident Score')
        st.plotly_chart(fig1)
        
        # Plot positive vs. concise
        fig2 = px.scatter(filtered_df, x='avg_positive', y='avg_concise', 
                        color='dominant_emotion_top1', 
                        title='Positive vs. Concise Score')
        st.plotly_chart(fig2)

        # Additional Plot: Speech Speed vs. Hesitant
        fig3 = px.scatter(filtered_df, x='avg_speech_speed', y='avg_hesitant', 
                        color='dominant_emotion_top1', 
                        title='Speech Speed vs. Hesitant Score')
        st.plotly_chart(fig3)
    # Call the function
    analyze_emotions_and_attributes_interactive(df)


    def composite_score_analysis(df):
        st.write("### Composite Score for Candidate Ranking:")
        
        # Create a composite score based on key attributes including speech speed and hesitation
        df['composite_score'] = (df['avg_positive'] + df['avg_confident'] + 
                                df['avg_enthusiastic'] - df['avg_hesitant'] - 
                                abs(df['avg_speech_speed'] - 3))  # Assuming normal speech speed is around 3
        
        # Sort by composite score
        ranked_df = df.sort_values(by='composite_score', ascending=False)
        
        st.write("#### Ranked Candidates by Composite Score:")
        st.write(ranked_df[['Unnamed: 0', 'composite_score']])

    # Call the function
    composite_score_analysis(df)






    # Dominant emotion counts
    st.write("### Dominant Emotion Counts:")
    fig = px.bar(df['dominant_emotion_top1'].value_counts(), title="Dominant Emotion Counts")
    st.plotly_chart(fig)

    # Pairplot equivalent with Plotly
    st.write("### Scatter Plot Matrix:")
    fig = px.scatter_matrix(df[['avg_positive', 'avg_negative', 'avg_neutral', 'avg_confident', 'avg_hesitant']])
    st.plotly_chart(fig)

    # # Boxplot for analyzing spread and outliers
    # st.write("### Boxplots for Outliers Detection:")
    # for feature in features:
    #     if df[feature].dtype in ['float64', 'int64']:
    #         st.write(f"#### {feature} Boxplot:")
    #         fig = px.box(df, y=feature, title=f'Boxplot of {feature}')
    #         st.plotly_chart(fig)
    
    # Distribution of emotions
    st.header("Distribution of Emotions")
    emotion_cols = ['avg_positive', 'avg_negative', 'avg_neutral']
    fig = make_subplots(rows=1, cols=3, subplot_titles=emotion_cols)
    for i, col in enumerate(emotion_cols):
        fig.add_trace(go.Histogram(x=df[col], name=f'Distribution of {col}', nbinsx=30), row=1, col=i+1)
    fig.update_layout(title_text="Distribution of Emotions", showlegend=False)
    st.plotly_chart(fig)

    # Boxplot of confident vs hesitant
    st.header("Confidence vs Hesitation")
    fig = px.box(df[['avg_confident', 'avg_hesitant']], title="Confidence vs Hesitation")
    st.plotly_chart(fig)

    # Time series analysis of speech speed
    st.header("Speech Speed Over Time")
    fig = px.line(df, x=df.index, y='avg_speech_speed', title='Speech Speed Over Time', labels={'x':'Video Index'})
    st.plotly_chart(fig)

    # Dominant emotion analysis
    st.header("Dominant Emotion Analysis")
    emotion_counts = df['dominant_emotion_top1'].value_counts()
    fig = px.bar(emotion_counts, title="Distribution of Dominant Emotions")
    st.plotly_chart(fig)

    # Statistical tests
    st.header("Statistical Tests")

    # T-test between confident and hesitant
    t_stat, p_value = stats.ttest_ind(df['avg_confident'], df['avg_hesitant'])
    st.write(f"T-test between confident and hesitant: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    # ANOVA for emotions across different speech speeds
    df['speed_category'] = pd.qcut(df['avg_speech_speed'], q=3, labels=['Slow', 'Medium', 'Fast'])
    f_stat, p_value = stats.f_oneway(df[df['speed_category'] == 'Slow']['avg_positive'],
                                     df[df['speed_category'] == 'Medium']['avg_positive'],
                                     df[df['speed_category'] == 'Fast']['avg_positive'])
    st.write(f"ANOVA for positive emotion across speech speeds: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")

    # Feature importance
    st.header("Feature Importance")

    X = df[['avg_positive', 'avg_negative', 'avg_neutral', 'avg_confident', 'avg_hesitant', 'avg_concise', 'avg_enthusiastic']]
    y = df['avg_speech_speed']
    model = RandomForestRegressor()
    model.fit(X, y)
    importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    importances = importances.sort_values('importance', ascending=False)
    fig = px.bar(importances, x='importance', y='feature', orientation='h', title="Feature Importance for Speech Speed")
    st.plotly_chart(fig)

    # Interactive widget for exploring relationships
    st.header("Explore Relationships")
    x_axis = st.selectbox("Choose X-axis", df.columns)
    y_axis = st.selectbox("Choose Y-axis", df.columns)
    fig = px.scatter(df, x=x_axis, y=y_axis, title=f'{x_axis} vs {y_axis}')
    st.plotly_chart(fig)




def main():
    if data_type == "Emotion Data":
        option = st.sidebar.selectbox("Select Student Option", range(1, 11))
        emotion_data_analysis(option)
        gaze_data_analysis(option)
    elif data_type == "Transcript Data":
        option = st.sidebar.selectbox("Select Transcript Option", range(1, 11))
        transcript_data_analysis(option)
    elif data_type == "Combined Emotion Analysis":
        combined_emotion_analysis()
    elif data_type== "Combined Gaze Data":
        combined_gaze_analysis()
    elif data_type== "final":
        show_final_analysis()
    


if __name__ == '__main__':
    main()
