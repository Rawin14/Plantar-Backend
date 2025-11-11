import { FastifyInstance } from 'fastify';
import { supabase, supabaseClient } from '../services/supabase';
import { z } from 'zod';

// Validation schemas
const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
  nickname: z.string().min(2)
});

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string()
});

const socialLoginSchema = z.object({
  provider: z.enum(['google', 'apple', 'facebook']),
  id_token: z.string()
});

export default async function authRoutes(fastify: FastifyInstance) {
  
  // Register
  fastify.post('/register', async (request, reply) => {
    try {
      const body = registerSchema.parse(request.body);

      // Create user
      const { data: authData, error: authError } = 
        await supabaseClient.auth.signUp({
          email: body.email,
          password: body.password
        });

      if (authError) {
        return reply.code(400).send({ error: authError.message });
      }

      if (!authData.user) {
        return reply.code(400).send({ error: 'Failed to create user' });
      }

      // Create profile
      const { error: profileError } = await supabase
        .from('profiles')
        .insert({
          id: authData.user.id,
          nickname: body.nickname
        });

      if (profileError) {
        console.error('Profile creation error:', profileError);
        return reply.code(500).send({ 
          error: 'User created but profile failed' 
        });
      }

      return {
        success: true,
        user: authData.user,
        session: authData.session
      };

    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.code(400).send({ 
          error: 'Validation error', 
          details: error.errors 
        });
      }
      throw error;
    }
  });

  // Login
  fastify.post('/login', async (request, reply) => {
    try {
      const body = loginSchema.parse(request.body);

      const { data, error } = await supabaseClient.auth.signInWithPassword({
        email: body.email,
        password: body.password
      });

      if (error) {
        return reply.code(401).send({ error: error.message });
      }

      return {
        success: true,
        user: data.user,
        session: data.session
      };

    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.code(400).send({ 
          error: 'Validation error', 
          details: error.errors 
        });
      }
      throw error;
    }
  });

  // Social Login
  fastify.post('/social', async (request, reply) => {
    try {
      const body = socialLoginSchema.parse(request.body);

      const { data, error } = await supabaseClient.auth.signInWithIdToken({
        provider: body.provider,
        token: body.id_token
      });

      if (error) {
        return reply.code(401).send({ error: error.message });
      }

      // Check if profile exists
      const { data: profile } = await supabase
        .from('profiles')
        .select()
        .eq('id', data.user.id)
        .single();

      // Create profile if doesn't exist
      if (!profile) {
        const nickname = 
          data.user.user_metadata?.full_name || 
          data.user.email?.split('@') || 
          'User';

        await supabase
          .from('profiles')
          .insert({
            id: data.user.id,
            nickname
          });
      }

      return {
        success: true,
        user: data.user,
        session: data.session
      };

    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.code(400).send({ 
          error: 'Validation error', 
          details: error.errors 
        });
      }
      throw error;
    }
  });

  // Logout
  fastify.post('/logout', async (request, reply) => {
    try {
      const authHeader = request.headers.authorization;
      
      if (!authHeader) {
        return reply.code(400).send({ error: 'No token provided' });
      }

      const token = authHeader.replace('Bearer ', '');

      const { error } = await supabase.auth.admin.signOut(token);

      if (error) {
        return reply.code(500).send({ error: error.message });
      }

      return { success: true };

    } catch (error) {
      throw error;
    }
  });

  // Get current user
  fastify.get('/me', {
    preHandler: async (request, reply) => {
      const authHeader = request.headers.authorization;
      
      if (!authHeader) {
        return reply.code(401).send({ error: 'No token provided' });
      }

      const token = authHeader.replace('Bearer ', '');
      const { data: { user }, error } = await supabase.auth.getUser(token);

      if (error || !user) {
        return reply.code(401).send({ error: 'Invalid token' });
      }

      (request as any).user = user;
    }
  }, async (request) => {
    const user = (request as any).user;

    // Get profile
    const { data: profile } = await supabase
      .from('profiles')
      .select()
      .eq('id', user.id)
      .single();

    return {
      user,
      profile
    };
  });
}