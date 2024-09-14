import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Set page configuration
st.set_page_config(layout="wide", page_title="Emotion and Transcript Data Dashboard")

# Sidebar
st.sidebar.title("Navigation")
data_type = st.sidebar.radio("Select Data Type", [
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



if data_type == "Emotion Data":
    option = st.sidebar.selectbox("Select Student", range(1, 11))
    
    # Load data
    @st.cache_data
    def load_emotion_data(option):
        emotion_df = pd.read_csv(f"emotion_data/{option}/emotion.csv")
        gaze_df = pd.read_csv(f"emotion_data/{option}/gaze.csv")
        return emotion_df, gaze_df
    
    emotion_df, gaze_df = load_emotion_data(option)
    
    
    
    # Main content
    st.title(f"Emotion Data Analysis - Student {option}")
    

    st.subheader("Emotion Data Sample")
    st.write(emotion_df.head())


    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(emotion_df.describe())
    
    # # Pair Plot to explore relationships between emotions
    # st.subheader("Pair Plot of Emotions")
    # fig = px.scatter_matrix(emotion_df["Dominatant_emotion"], title="Pair Plot of Emotions")
    # st.plotly_chart(fig)

    # Emotion distribution countplot
    st.subheader("Emotion Distribution")
    fig = px.histogram(emotion_df, x='dominant_emotion', title="Distribution of Dominant Emotions")
    st.plotly_chart(fig)
    
    # Emotion intensity heatmap
    st.subheader("Emotion Intensity Heatmap")
    emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    fig = px.imshow(emotion_df[emotion_columns].transpose(), 
                    labels=dict(x="Frame", y="Emotion", color="Intensity"),
                    title="Emotion Intensity Heatmap")
    st.plotly_chart(fig)
    
    
    # Emotion Transition Matrix (shift dominant_emotion by 1 and crosstab)
    st.subheader("Emotion Transition Matrix")
    emotion_df['prev_emotion'] = emotion_df['dominant_emotion'].shift(1)
    transition_matrix = pd.crosstab(emotion_df['prev_emotion'], emotion_df['dominant_emotion'], normalize='index')
    fig = px.imshow(transition_matrix, labels=dict(x="To", y="From", color="Transition Probability"),
                    title="Emotion Transition Matrix")
    st.plotly_chart(fig)                 

    # Emotion radar chart
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
    
    # hello
    # Plot emotion intensities over time
    st.subheader("Emotion Intensities Over Time")
    emotion_intensities = emotion_df.melt(id_vars=['movie_id', 'image_seq'], 
                                          value_vars=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
                                          var_name='emotion', value_name='intensity')
    fig = px.line(emotion_intensities, x='image_seq', y='intensity', color='emotion', title="Emotion Intensities Over Time")
    st.plotly_chart(fig)
    
    # Correlation heatmap
    st.subheader("Emotion Correlation Heatmap")
    corr = emotion_df[['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Emotion Correlation Heatmap")
    st.plotly_chart(fig)
    
    # Boxplot of emotion intensities
    st.subheader("Boxplot of Emotion Intensities")
    fig = px.box(emotion_intensities, x='emotion', y='intensity', title="Distribution of Emotion Intensities")
    st.plotly_chart(fig)

    # zello
    
    # 3D scatter plot of top 3 emotions
    st.subheader("3D Scatter Plot of Top 3 Emotions")
    top_3_emotions = emotion_df[emotion_columns].mean().nlargest(3).index.tolist()
    fig = px.scatter_3d(emotion_df, x=top_3_emotions[0], y=top_3_emotions[1], z=top_3_emotions[2],
                        color='dominant_emotion', title=f"3D Scatter of {', '.join(top_3_emotions)}")
    st.plotly_chart(fig)

    # EDA for Gaze Data
    st.header("Gaze Data Analysis")
    
    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(gaze_df.describe())
    
    # Plot gaze and blink over time
    st.subheader("Gaze and Blink Over Time")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Gaze Over Time", "Blink Over Time"))
    fig.add_trace(go.Scatter(x=gaze_df['image_seq'], y=gaze_df['gaze'], mode='lines', name='Gaze'), row=1, col=1)
    fig.add_trace(go.Scatter(x=gaze_df['image_seq'], y=gaze_df['blink'], mode='lines', name='Blink'), row=2, col=1)
    fig.update_layout(height=600, title_text="Gaze and Blink Over Time")
    st.plotly_chart(fig)
    
    # Plot eye offset distribution
    st.subheader("Eye Offset Distribution")
    fig = px.histogram(gaze_df, x='eye_offset', title="Eye Offset Distribution")
    st.plotly_chart(fig)
    
    # Scatter plot of gaze vs blink
    st.subheader("Gaze vs Blink Scatter Plot")
    fig = px.scatter(gaze_df, x='gaze', y='blink', color='eye_offset',
                     title="Gaze vs Blink (colored by Eye Offset)")
    st.plotly_chart(fig)
    
    # Time series decomposition of gaze data
    st.subheader("Gaze Time Series Decomposition")
    from statsmodels.tsa.seasonal import seasonal_decompose
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

elif data_type == "Transcript Data":
    option = st.sidebar.selectbox("Select Option", range(1, 11))
    
    # Load data
    @st.cache_data
    def load_transcript_data(option):
        transcript_df = pd.read_csv(f"transcript_data/{option}.csv")
        return transcript_df
    
    transcript_df = load_transcript_data(option)
    
    # Main content
    st.title(f"Transcript Data Analysis - Option {option}")
    
    # Display transcript data sample
    st.subheader("Transcript Data Sample")
    st.write(transcript_df.head())
    
    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(transcript_df.describe())
    
    # Speech duration analysis
    st.subheader("Speech Duration Analysis")
    transcript_df['duration'] = transcript_df['end'] - transcript_df['start']
    fig = px.histogram(transcript_df, x='duration', 
                       title="Distribution of Speech Durations")
    st.plotly_chart(fig)
    
    # Sentiment analysis
    st.subheader("Sentiment Analysis")
    sentiment_cols = ['positive', 'negative', 'neutral']
    fig = px.bar(transcript_df, x=transcript_df.index, y=sentiment_cols, 
                 title="Sentiment Scores for Each Speech Segment")
    st.plotly_chart(fig)
    

    # Dual Axis Plot: Speech Speed and Confidence Over Time
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


    # Speech characteristics radar chart
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


elif data_type == "Combined Emotion Analysis":
    st.sidebar.title("Combined Emotion Data Analysis")
    if st.sidebar.button(label="EDA FOR ALL EMOTION DATA"):
        st.title("Combined Emotion Data Analysis")

        # Load all emotion data
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


elif data_type == "Combined Gaze Data":
    st.sidebar.title("Combined GAZE Data Analysis")
    if st.sidebar.button(label="EDA FOR ALL GAZE DATA"):
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


elif data_type == "Emotion with gaze and transcript analysis":
    st.sidebar.title("Emotion with gaze and transcript analysis")
    if st.sidebar.button(label="Emotion with gaze and transcript analysis"):
        st.title("Combined Emotion, Gaze, and Transcript Analysis")

        # Load all emotion, gaze, and transcript data
        all_emotion_data = load_all_emotion_data()
        all_gaze_data = load_all_gaze_data()
        all_transcript_data = load_all_transcript_data()

        # Merge datasets on 'option' and 'image_seq'
        combined_data = pd.merge(all_emotion_data, all_gaze_data, on=['option', 'image_seq'])
        combined_data = pd.merge(combined_data, all_transcript_data, on=['option'])

        st.subheader("Combined Data Sample")
        st.write(combined_data.head())

        # Sentiment vs Emotion Scatter
        st.subheader("Sentiment vs. Emotion Scatter Plot")
        fig = px.scatter(combined_data, x='positive', y='happy', 
                         color='dominant_emotion', size='gaze',
                         title="Positive Sentiment vs. Happy Emotion (Size by Gaze)")
        st.plotly_chart(fig)

        # Blink Rate vs. Speech Speed
        st.subheader("Blink Rate vs. Speech Speed")
        fig = px.scatter(combined_data, x='speech_speed', y='blink', 
                         color='dominant_emotion', 
                         title="Blink Rate vs. Speech Speed (colored by Dominant Emotion)")
        st.plotly_chart(fig)

        # Gaze vs. Sentiment Time Series
        st.subheader("Gaze and Sentiment Over Time")
        fig = px.line(combined_data, x='image_seq', y=['gaze', 'positive', 'negative', 'neutral'],
                      title="Gaze and Sentiment Scores Over Time")
        st.plotly_chart(fig)

        st.subheader("Emotion vs Gaze vs Speech Speed")
        fig = px.scatter_3d(combined_data, x='gaze', y='speech_speed', z='happy', 
                            color='dominant_emotion', size='blink',
                            title="3D Scatter: Gaze vs Speech Speed vs Happy Emotion")
        st.plotly_chart(fig)

        st.subheader("Emotion and Gaze Correlation Heatmap")
        corr_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'gaze', 'blink', 'eye_offset']
        corr_matrix = combined_data[corr_columns].corr()
        fig = px.imshow(corr_matrix, title="Correlation Heatmap: Emotions and Gaze")
        st.plotly_chart(fig)

        st.subheader("Speech Characteristics vs Dominant Emotion")
        speech_chars = ['confident', 'hesitant', 'concise', 'enthusiastic']
        speech_emotion_data = combined_data.groupby('dominant_emotion')[speech_chars].mean().reset_index()
        fig = px.bar(speech_emotion_data, x='dominant_emotion', y=speech_chars, 
                    title="Average Speech Characteristics by Dominant Emotion")
        st.plotly_chart(fig)

        st.subheader("Gaze and Emotion Time Series")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            subplot_titles=("Gaze Over Time", "Dominant Emotion Over Time"))
        fig.add_trace(go.Scatter(x=combined_data['image_seq'], y=combined_data['gaze'], mode='lines', name='Gaze'), 
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=combined_data['image_seq'], y=combined_data['dominant_emotion'], mode='lines', name='Dominant Emotion'), 
                    row=2, col=1)
        fig.update_layout(height=600, title_text="Gaze and Dominant Emotion Over Time")
        st.plotly_chart(fig)
        

# Footer
st.sidebar.markdown("---")

def show_final_analysis():
    st.title("Final Dataset")
    st.subheader("Merged Data: Emotion, Gaze, and Transcript")
    
    # Replace with your actual DataFrame
    df_final = ...  # Load or process your DataFrame here
    st.write(df_final)

# Main code for Streamlit app
def main():
    st.sidebar.markdown("---")
    st.sidebar.title("Analysis Options")
    
    if st.sidebar.button(label="Final Analysis"):
        show_final_analysis()

if __name__ == "__main__":
    main()
    
