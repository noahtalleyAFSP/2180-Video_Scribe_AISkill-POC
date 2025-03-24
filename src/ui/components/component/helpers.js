import axios from 'axios';  
  
// Base URL for your Flask API  
const BASE_URL = 'http://localhost:5000';  
  
// Function to login a user  
export const loginUser = async (email, password) => {  
  try {  
    const response = await axios.post(`${BASE_URL}/login`, {  
      email,  
      password  
    });  
    return response.data; // Contains user_id  
  } catch (error) {  
    console.error('Login error:', error);  
    return { error: error.response.data.error };  
  }  
};  
  
// Function to register a new user  
export const registerUser = async (email, password,username,name) => {  
  try {  
    const response = await axios.post(`${BASE_URL}/register`, {  
      email,  
      password,
      username,
      name  
    });  
    return response.data; // Contains user_id  
  } catch (error) {  
    console.error('Registration error:', error);  
    return { error: error.response.data.error };  
  }  
};  
  
// Function to upload a video  
export const uploadVideo = async (videoUrl, videoName, userId) => {  
  try {  
    const response = await axios.post(`${BASE_URL}/upload`, {  
      video_url: videoUrl,  
      video_name: videoName,  
      user_id: userId  
    });  
    return response.data; // Contains message  
  } catch (error) {  
    console.error('Upload error:', error);  
    return { error: error.response.data.error };  
  }  
};  
  
// Function to get data by user ID  
export const getDataById = async (userId) => {  
  try {  
    const response = await axios.get(`${BASE_URL}/data/${userId}`);  
    return response.data; // Contains the data  
  } catch (error) {  
    console.error('Get data error:', error);  
    return { error: error.response.data.error };  
  }  
};  
  
// Function to overwrite data by data ID  
export const overwriteDataById = async (dataId, name, newData) => {  
  try {  
    const response = await axios.put(`${BASE_URL}/data/${dataId}`, {  
      name,  
      data: newData  
    });  
    return response.data; // Contains message  
  } catch (error) {  
    console.error('Overwrite data error:', error);  
    return { error: error.response.data.error };  
  }  
};  
  
// Function to delete data by data ID  
export const deleteDataById = async (dataId) => {  
  try {  
    const response = await axios.delete(`${BASE_URL}/data/${dataId}`);  
    return response.data; // Contains message  
  } catch (error) {  
    console.error('Delete data error:', error);  
    return { error: error.response.data.error };  
  }  
};  
  
// Function to find data by name  
export const findDataByName = async (name) => {  
  try {  
    const response = await axios.get(`${BASE_URL}/data/find`, {  
      params: { name }  
    });  
    return response.data; // Contains the data  
  } catch (error) {  
    console.error('Find data error:', error);  
    return { error: error.response.data.error };  
  }  
};  
