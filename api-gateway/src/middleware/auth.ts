import { FastifyRequest, FastifyReply } from 'fastify';
import { supabase } from '../services/supabase';

export async function authenticate(
  request: FastifyRequest,
  reply: FastifyReply
) {
  try {
    const authHeader = request.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return reply.code(401).send({
        error: 'Missing or invalid authorization header'
      });
    }

    const token = authHeader.replace('Bearer ', '');

    // Verify token with Supabase
    const { data: { user }, error } = await supabase.auth.getUser(token);

    if (error || !user) {
      return reply.code(401).send({
        error: 'Invalid token'
      });
    }

    // Attach user to request
    (request as any).user = user;

  } catch (error) {
    console.error('Auth middleware error:', error);
    return reply.code(500).send({
      error: 'Authentication failed'
    });
  }
}