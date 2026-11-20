import {createRoot} from 'react-dom/client';import App from './App';import './style.css';import './studio-theme.css';import './simulation-studio.css';
const root=document.getElementById('root');if(!root)throw Error('Không tìm thấy vùng hiển thị ứng dụng.');createRoot(root).render(<App/>);
