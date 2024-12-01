# GitHub Repository Analysis with Pinecone and LangChain - RepoChat
<img src="https://img.shields.io/badge/-Solo Project-f2336f?&style=for-the-badge&logoColor=white" />

## Objective

This project implements a codebase search and analysis tool leveraging Pinecone for vector storage and LangChain for building conversational AI capabilities. It allows users to query GitHub repositories for insights into the codebase and provides intelligent answers by analyzing code files and metadata.

The project includes embedding repositories into Pinecone, cloning GitHub repositories, and querying via a conversational interface using HuggingFace embeddings and OpenAI models.

**To view the app, <a href="https://edom-repochat.streamlit.app/"> Click-here</a>.** 

### Skills Learned

- Proficiency in Python and its libraries for building end-to-end applications.
- Integration of **Streamlit** for building interactive web applications.
- Leveraging **Pinecone** for vector storage and efficient similarity searches.
- Familiarity with **LangChain** for conversational AI and document processing.
- Use of **HuggingFace** models for generating embeddings from code and natural language text.
- Working with **GitHub APIs** and automating repository cloning and processing.
- Applying best practices for managing namespaces in Pinecone and preventing reprocessing.
- Building robust error handling and debugging mechanisms for processing large repositories.
- Knowledge of **open-source repository structures** and effective handling of diverse file formats.

### Tools Used
<div>
  <img src="https://img.shields.io/badge/-Python-3776AB?&style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?&style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/-Pinecone-2563eb?&style=for-the-badge&logo=pinecone&logoColor=white" />
  <img src="https://img.shields.io/badge/-HuggingFace-FEA053?&style=for-the-badge&logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/-LangChain-10A8E0?&style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/-OpenAI-412991?&style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/-GitHub-181717?&style=for-the-badge&logo=github&logoColor=white" />
</div>

## Steps and Features

### Key Features
- **GitHub Repository Cloning:** The application clones public repositories to analyze the content locally.
- **Embedding and Storage:** Code files are embedded using HuggingFace sentence transformers and stored in Pinecone for similarity searches.
- **Query Processing:** Users can query the repository via a conversational interface. LangChain generates responses based on embeddings and context.
- **Namespace Management:** Prevents duplicate processing by tracking repository namespaces in Pinecone.
- **Support for Multiple File Types:** Handles Python, JavaScript, TypeScript, Java, and more, ensuring versatility.

### Workflow
1. **Clone Repository:** Input a GitHub repository link to clone the codebase locally.
2. **Extract File Content:** Supported files are scanned, and their content is read.
3. **Embed and Store:** HuggingFace embeddings are generated and stored in Pinecone.
4. **Query Codebase:** Enter questions to retrieve relevant information about the codebase.
5. **Chat Threads:** Manage multiple chat sessions for querying different repositories.

### Screenshots
<img width="1440" alt="Screenshot 2024-11-30 at 7 49 19 PM" src="https://github.com/user-attachments/assets/2e6bd372-2617-4437-bf75-6c2045c32cfe">
<img width="1440" alt="Screenshot 2024-11-30 at 7 49 19 PM" src="https://github.com/user-attachments/assets/a911c511-ade5-4e3b-8165-ecf77ec5c56d">




## To Execute



## Future Enhancements
- **Authentication:** Add support for private GitHub repositories using user credentials.
- **Enhanced Search:** Leverage advanced language models for better query understanding and responses.
- **Visualization:** Include visualizations of repository structure and file relationships.
