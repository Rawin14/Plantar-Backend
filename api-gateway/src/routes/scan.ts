import { FastifyInstance } from 'fastify';
import { authenticate } from '../middleware/auth';
import { supabase } from '../services/supabase';
import { z } from 'zod';
import fetch from 'node-fetch';

const createScanSchema = z.object({
  foot_side: z.enum(['left', 'right']),
  images_url: z.array(z.string().url()).min(3)
});

export default async function scanRoutes(fastify: FastifyInstance) {
  
  // Get all user scans
  fastify.get('/', {
    preHandler: authenticate
  }, async (request) => {
    const user = (request as any).user;

    const { data, error } = await supabase
      .from('foot_scans')
      .select(`
        *,
        shoe_recommendations (*)
      `)
      .eq('user_id', user.id)
      .order('created_at', { ascending: false });

    if (error) {
      throw new Error(error.message);
    }

    return data;
  });

  // Get single scan
  fastify.get('/:id', {
    preHandler: authenticate
  }, async (request, reply) => {
    const user = (request as any).user;
    const { id } = request.params as { id: string };

    const { data, error } = await supabase
      .from('foot_scans')
      .select(`
        *,
        shoe_recommendations (*)
      `)
      .eq('id', id)
      .eq('user_id', user.id)
      .single();

    if (error) {
      return reply.code(404).send({ error: 'Scan not found' });
    }

    return data;
  });

  // Create scan
  fastify.post('/', {
    preHandler: authenticate
  }, async (request, reply) => {
    try {
      const user = (request as any).user;
      const body = createScanSchema.parse(request.body);

      // Create scan record
      const { data: scan, error: scanError } = await supabase
        .from('foot_scans')
        .insert({
          user_id: user.id,
          foot_side: body.foot_side,
          images_url: body.images_url,
          status: 'processing'
        })
        .select()
        .single();

      if (scanError) {
        return reply.code(500).send({ error: scanError.message });
      }

      // Trigger ML processing
      const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://localhost:8000';
      
      fetch(`${mlServiceUrl}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scan_id: scan.id,
          image_urls: body.images_url
        })
      }).catch(err => {
        console.error('Failed to trigger ML processing:', err);
      });

      return scan;

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

  // Delete scan
  fastify.delete('/:id', {
    preHandler: authenticate
  }, async (request, reply) => {
    const user = (request as any).user;
    const { id } = request.params as { id: string };

    const { error } = await supabase
      .from('foot_scans')
      .delete()
      .eq('id', id)
      .eq('user_id', user.id);

    if (error) {
      return reply.code(500).send({ error: error.message });
    }

    return { success: true };
  });
}