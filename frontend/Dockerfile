# rag-react-app/Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy all source code
COPY . .

# Build the app (optional, for production)
# RUN npm run build

EXPOSE 3000

# Start the development server
CMD ["npm", "start"]