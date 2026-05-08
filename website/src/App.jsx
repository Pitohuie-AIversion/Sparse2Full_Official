import React, { Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import './i18n'; // initialize i18n

const Home = React.lazy(() => import('./pages/Home'));
const Docs = React.lazy(() => import('./pages/Docs'));
const About = React.lazy(() => import('./pages/About'));
const Contact = React.lazy(() => import('./pages/Contact'));
const Login = React.lazy(() => import('./pages/Login'));
const Admin = React.lazy(() => import('./pages/Admin'));

function App() {
  return (
    <Router basename="/Sparse2Full_Official">
      <Suspense fallback={<div className="min-h-screen flex items-center justify-center text-slate-400">Loading...</div>}>
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/docs" element={<Docs />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/login" element={<Login />} />
            <Route path="/admin" element={<Admin />} />
          </Routes>
        </Layout>
      </Suspense>
    </Router>
  );
}

export default App;
