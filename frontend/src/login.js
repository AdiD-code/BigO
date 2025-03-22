import React, { useState } from "react";
import { Link } from "react-router-dom";

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Login attempt with", email, password);
    // Add authentication logic here
  };

  return (
    <div className="flex h-screen justify-center items-center bg-gradient-to-br from-red-400 to-orange-500">
      <div className="bg-white p-8 rounded-2xl shadow-lg w-96">
        <h2 className="text-3xl font-bold text-center text-gray-800 mb-6">Tufaan Ticket</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-gray-700">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
              required
            />
          </div>
          <div>
            <label className="block text-gray-700">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
              required
            />
          </div>
          <button
            type="submit"
            className="w-full bg-orange-600 text-white py-2 rounded-lg hover:bg-orange-700 transition"
          >
            Login
          </button>
        </form>
        <p className="text-center text-gray-600 mt-4">
          Don't have an account? <Link to="/signup" className="text-orange-600 font-semibold">Sign Up</Link>
        </p>
      </div>
    </div>
  );
};

export default Login;