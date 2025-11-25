import dotenv from 'dotenv';
// Load environment variables
dotenv.config();
import Fastify from 'fastify';
import cors from '@fastify/cors';
import multipart from '@fastify/multipart';

// Routes
import authRoutes from './routes/auth';
import profileRoutes from './routes/profile';
import scanRoutes from './routes/scan';
import shoeRoutes from './routes/shoes';


const fastify = Fastify({
  logger: {
    level: process.env.NODE_ENV === 'production' ? 'info' : 'debug'
  }
});

// Register plugins
fastify.register(cors, {
  origin: true,
  credentials: true
});

fastify.register(multipart, {
  limits: {
    fileSize: 100 * 1024 * 1024 // 100MB
  }
});

// Register routes
fastify.register(authRoutes, { prefix: '/api/auth' });
fastify.register(profileRoutes, { prefix: '/api/profile' });
fastify.register(scanRoutes, { prefix: '/api/scan' });
fastify.register(shoeRoutes, { prefix: '/api/shoes' });

// Health check
fastify.get('/health', async () => {
  return { 
    status: 'ok',
    timestamp: new Date().toISOString(),
    service: 'api-gateway'
  };
});

// Root
fastify.get('/', async () => {
  return {
    name: 'Plantar API Gateway',
    version: '1.0.0',
    endpoints: {
      auth: '/api/auth',
      profile: '/api/profile',
      scan: '/api/scan',
      shoes: '/api/shoes'
    }
  };
});

// Start server
const start = async () => {
  try {
    const port = parseInt(process.env.PORT || '4000');
    const host = process.env.HOST || '0.0.0.0';

    await fastify.listen({ port, host });
    
    console.log(`
🚀 API Gateway is running!
📍 URL: http://localhost:${port}
🌍 Environment: ${process.env.NODE_ENV || 'development'}
    `);

  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();