from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "FullData.csv")
df = pd.read_csv(csv_path)
MLData = df[["ind", "Tags"]]
vectorizer = CountVectorizer()
dataset_matrix = vectorizer.fit_transform(MLData['Tags'])

def recommend(skills_list):
    if isinstance(skills_list, list): 
        skills_text = " ".join(skills_list)
    else: 
        skills_text = str(skills_list)

    new_vector = vectorizer.transform([skills_text])
    similarities = cosine_similarity(new_vector, dataset_matrix).flatten()
    valid_indices = np.where(similarities >= 0.01)[0]
    if len(valid_indices) == 0: 
        return []
    else:
        top_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1][:5]]
        recommended = df.iloc[top_indices]
        return recommended[["job_title", "job_description"]].to_dict(orient="records")

def suggest(skills):
    print(f" DEBUG: suggest() called with skills: {skills}")
    
    if isinstance(skills, list):
        skills_text = " ".join(skills)
    else:
        skills_text = str(skills)

    new_vector = vectorizer.transform([skills_text])
    similarities = cosine_similarity(new_vector, dataset_matrix).flatten()
    valid_indices = np.where(similarities >= 0.1)[0]
    
    print(f" DEBUG: Found {len(valid_indices)} valid indices with similarity >= 0.1")
    
    if len(valid_indices) == 0:
        print(" DEBUG: No valid indices found, returning empty list")
        return []
    else:
        
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        if len(sorted_indices) <= 5:
            print("  DEBUG: Not enough matches, using all available indices")
            top_indices = sorted_indices 
        else:
            top_indices = sorted_indices[5:9]  
            
        print(f"🔍 DEBUG: Using indices: {top_indices}")
        
        if len(top_indices) == 0:
            print(" DEBUG: No indices after slicing, returning empty list")
            return []
            
        recommended = df.iloc[top_indices]
        recommended = recommended[["job_title", "job_description", "skills"]].to_dict(orient="records")
        
        print(f" DEBUG: Initial recommended list has {len(recommended)} items")

    wordfreq = {}
    unwanted_chars = ".,-_()"
    filter_words = {
        "and", "or", "e.g", "i.e", "etc", "with", "by", "in", "on", "of",
        "for", "at", "to", "from", "about", "the", "a", "an", "such", "as",
        "including", "among", "which", "while", "but", "if", "like", "because"
    }
    
    for i in recommended:
        words = i['skills'].split()
        for raw_word in words:
            
            word = raw_word.strip(unwanted_chars)
            if word.lower() in filter_words:  
                continue
            if word not in wordfreq:
                wordfreq[word] = 0 
            wordfreq[word] += 1
        
    sorted_list = dict(sorted(wordfreq.items(), key=lambda item: item[1], reverse=True))
    print(f" DEBUG: Word frequency dict has {len(sorted_list)} words")

    recommended_skills = []
    for word in sorted_list:
        if word not in skills:
            recommended_skills.append(word)
        if len(recommended_skills) >= 8: 
            break

    print(f" DEBUG: Found {len(recommended_skills)} recommended skills: {recommended_skills}")

    # Match skills 
    for i in range(len(recommended)):
        for j in recommended_skills[:]:  
            if j in recommended[i]["skills"]:
                recommended[i]["suggested_skill"] = j
                recommended_skills.remove(j)
                break
                

    final_recommended = []
    for item in recommended:
        if "suggested_skill" in item:  
            final_recommended.append(item)
    
    print(f" DEBUG: Final recommended list has {len(final_recommended)} items")
    for item in final_recommended:
        print(f"   - {item['job_title']} (suggested: {item['suggested_skill']})")
        
    return final_recommended

@app.route('/profile-setup', methods=['POST'])
def profile_setup():
    data = request.get_json()
    name = data.get("name")
    skills = data.get("skills", [])
    education = data.get("education")
    state = data.get("state")
    print(f" New profile: {name}, Skills: {skills}, Education: {education}, State: {state}")

    results = recommend(skills)
    partial_results = suggest(skills)
    
    print(f"Recommendations: {len(results)} items")
    for r in results:
        print(f"   - {r['job_title']}")
        
    print(f"🔄 Partial Matches: {len(partial_results)} items")
    for p in partial_results:
        print(f"   - {p['job_title']} (skill: {p.get('suggested_skill', 'N/A')})")
    
    if not results:
        return jsonify({
            "message": "Profile saved successfully, but no recommendations found.",
            "receivedSkills": skills,
            "recommendations": [],
            "partialMatches": partial_results 
        })
    
    return jsonify({
        "message": "Profile saved successfully",
        "receivedSkills": skills,
        "recommendations": results,
        "partialMatches": partial_results  
    })