import axios from 'axios';
import config from '../config';

const axiosInstance = axios.create({
  baseURL: config.apiBaseUrl,
  headers: {
    "ngrok-skip-browser-warning": "69420",
    "Access-Control-Allow-Origin": config.apiBaseUrl
  }
});

export default axiosInstance;