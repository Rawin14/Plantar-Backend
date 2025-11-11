import { FastifyInstance } from 'fastify';
import { authenticate } from '../middleware/auth';
import { supabase } from '../services/supabase';
import { z } from 'zod';

const updateProfileSchema = z.object({
  nickname: z.string().min(2).optional(),
  age: z.number().int().min(1).max(150).optional(),
  height: z.number().min(50).max(300).optional(),
  weight: z.number().min(20).max(500).optional(),
  gender: z.enum(['male', 'female', 'other']).optional(),
  birthdate: z.string().optional()
});

export default async function profileRoutes(fastify: FastifyInstance) {
  
  // Get profile
  fastify.get('/', {
    preHandler: authenticate
  }, async (request) => {
    const user = (request as any).user;

    const { data, error } = await supabase
      .from('profiles')
      .select()
      .eq('id', user.id)
      .single();

    if (error) {
      throw new Error(error.message);
    }

    return data;
  });

  // Update profile
  fastify.patch('/', {
    preHandler: authenticate
  }, async (request, reply) => {
    try {
      const user = (request as any).user;
      const body = updateProfileSchema.parse(request.body);

      const { data, error } = await supabase
        .from('profiles')
        .update(body)
        .eq('id', user.id)
        .select()
        .single();

      if (error) {
        return reply.code(500).send({ error: error.message });
      }

      return data;

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
}