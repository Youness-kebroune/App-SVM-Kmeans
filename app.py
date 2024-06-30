import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np

# Set page title
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    """
    <style>
    .stApp {
        background: url('https://w0.peakpx.com/wallpaper/249/226/HD-wallpaper-simple-luxury-clean-dark-gold-golden-luxury-metallic-pattern-shiny-simple.jpg') no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    
    .logo {
        position: absolute;
        top: 160px;
        right: 10px;
        width: 300px; 
    }

    .custom-title {
        color: #BFBFBF;
        font-size: 3em; 
        font-family: 'Arial', sans-serif;
        font-weight: bold; 
        text-align: center; 
        margin-bottom: 20px; 
    }
    .names-board {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        border: 2px solid #D99E30; /* Change this to your desired border color */
        padding: 10px;
        width: 300px;
        margin: 20px;
        background-color: ;
        border-radius: 150px; 
    }
    .names-board p {
        margin: 5px 0;
    }
    </style>
    <img src="https://www.guide-metiers.ma/wp-content/uploads/2019/03/fmpmarrakech.couleur-min-206x206.png" class="logo">
    """,
    unsafe_allow_html=True
)

# Set page title with custom CSS class
st.markdown('<p class="custom-title">Explore & Classify: K-means Clustering and SVM Models</p>', unsafe_allow_html=True)

st.markdown("""
*Created by:*

<div class="names-board">
  <p><strong>Ait Kebroune Youness</strong></p>
  <p><strong>El Basit Mouad</strong></p>
  <p><strong>Abdennacer Abhouss</strong></p>
  <p><strong>Abia Maryam</strong></p>
</div>
""", unsafe_allow_html=True)

# Step 2: Upload CSV file
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("## Data Preview")
    st.write(data.head())

    # Step 3: Visualize Data
    if st.checkbox("Show Data Visualization"):
        st.write("### Data Visualization")
        sns.pairplot(data)
        st.pyplot()

    # Step 4: Model selection
    st.write("## Choose a Model")
    model_option = st.selectbox("Select Model", ("K-means", "SVM"))

    if model_option == "K-means":
        st.write("### K-means Clustering")
        html = """
<div style="background-color: #f0f0f0; padding: 20px; border-radius: 150px; border: 1px solid #ccc;">
    K-means clustering groups data into \(k\) clusters by assigning points to the nearest centroid, recalculating centroids as the mean of points in each cluster, and repeating until centroids stabilize.
</div>
"""
        # Select columns to use
        selected_columns = st.multiselect("Select Features Columns", data.columns)

        if selected_columns:
            # K-means method selection
            init_method = st.selectbox("Initialization Method", ("k-means++", "random"))
            
            # Option to choose fixed number of clusters or interval
            cluster_option = st.radio("Select Cluster Option", ("Fixed Number", "Interval"))

            if cluster_option == "Fixed Number":
                k_value = st.number_input("Enter the number of clusters (K)", min_value=1, max_value=20, value=3)
            elif cluster_option == "Interval":
                start_k = st.number_input("Enter the start of the cluster range", min_value=1, max_value=20, value=1)
                end_k = st.number_input("Enter the end of the cluster range", min_value=1, max_value=20, value=10)
                
                if st.button("Plot Inertia Curve"):
                    inertia_values = []
                    for k in range(start_k, end_k + 1):
                        kmeans = KMeans(n_clusters=k, init=init_method)
                        kmeans.fit(data[selected_columns].fillna(data[selected_columns].mean()))
                        inertia_values.append(kmeans.inertia_)
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(start_k, end_k + 1), inertia_values, marker='o')
                    plt.title('Inertia vs Number of Clusters')
                    plt.xlabel('Number of Clusters')
                    plt.ylabel('Inertia')
                    st.pyplot()
                    
                    k_value = st.number_input("Enter the optimal number of clusters (K) based on the inertia curve", min_value=start_k, max_value=end_k, value=start_k)
            
            if st.button("Run K-means"):
                # Preprocess the data
                numeric_data = data[selected_columns].dropna()

                if numeric_data.empty:
                    st.write("No numeric data available for clustering.")
                else:
                    kmeans = KMeans(n_clusters=k_value, init=init_method)
                    kmeans.fit(numeric_data.fillna(numeric_data.mean()))
                    labels = kmeans.labels_
                    inertia = kmeans.inertia_
                    
                    st.write("### Clustering Results")
                    st.write(f"Inertia: {inertia}")
                    st.write(f"Cluster Centers: {kmeans.cluster_centers_}")
                    
                    numeric_data['Cluster'] = labels
                    pca = PCA(2)
                    pca_data = pca.fit_transform(numeric_data)
                    plt.figure(figsize=(10, 6))
                    plt.scatter(pca_data[:,0], pca_data[:,1], c=labels, cmap='viridis')
                    plt.title('K-means Clustering')
                    st.pyplot()

    elif model_option == "SVM":
        st.write("### Support Vector Machine (SVM) Classification")
        st.write("### Support Vector Machines (SVM) classify data by finding the hyperplane that best separates different classes, maximizing the margin between them.")

        # Selection of columns to use
        selected_columns = st.multiselect("Select Features Columns", data.columns)
        target_column = st.selectbox("Select Target Column", data.columns)

        # SVM kernel selection
        kernel_option = st.selectbox("Select Kernel", ("rbf", "linear", "poly"))

        # Display and input parameters based on kernel choice
        if kernel_option == "rbf":
            gamma = st.number_input("Gamma", value=1.0)
            C = st.number_input("C", value=1.0)
        elif kernel_option == "linear":
            C = st.number_input("C", value=1.0)
        elif kernel_option == "poly":
            degree = st.number_input("Degree", value=3)
            C = st.number_input("C", value=1.0)

        if st.button("Run SVM"):
            if not selected_columns or target_column in selected_columns:
                st.write("Please select valid feature columns excluding the target column.")
            else:
                X = data[selected_columns]
                y = data[target_column]

                # Encode categorical variables
                X = pd.get_dummies(X, drop_first=True)

                # Replace missing values with mean
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())

                # Initialize SVM with selected kernel and parameters
                if kernel_option == "rbf":
                    svm_model = SVC(kernel="rbf", gamma=gamma, C=C)
                elif kernel_option == "linear":
                    svm_model = SVC(kernel="linear", C=C)
                elif kernel_option == "poly":
                    svm_model = SVC(kernel="poly", degree=degree, C=C)

                svm_model.fit(X, y)
                predictions = svm_model.predict(X)

                st.write("### SVM Classification Result")
                unique_classes, counts = np.unique(predictions, return_counts=True)
                for cls, count in zip(unique_classes, counts):
                    st.write(f"Class {cls}: {count} instances")

                # PCA for visualization
                min_samples_features = min(X.shape[0], X.shape[1])
                if min_samples_features < 2:
                    st.write("Not enough samples or features to perform PCA for visualization.")
                else:
                    pca = PCA(n_components=2)
                    pca_data = pca.fit_transform(X)
                    plt.figure(figsize=(10, 6))
                    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=predictions, cmap='viridis')
                    plt.title('SVM Classification')
                    plt.xlabel(selected_columns[0])
                    plt.ylabel(selected_columns[1])
                    st.pyplot()
