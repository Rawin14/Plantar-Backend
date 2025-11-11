import { FastifyInstance } from 'fastify';
import { supabase } from '../services/supabase';

export default async function shoeRoutes(fastify: FastifyInstance) {
  
  // Get all shoes
  fastify.get('/', async (request) => {
    const { 
      brand, 
      category, 
      min_price, 
      max_price,
      limit = 20,
      offset = 0
    } = request.query as any;

    let query = supabase
      .from('shoes')
      .select('*', { count: 'exact' });

    if (brand) {
      query = query.eq('brand', brand);
    }

    if (category) {
      query = query.eq('category', category);
    }

    if (min_price) {
      query = query.gte('price', parseFloat(min_price));
    }

    if (max_price) {
      query = query.lte('price', parseFloat(max_price));
    }

    const { data, error, count } = await query
      .range(offset, offset + limit - 1)
      .order('created_at', { ascending: false });

    if (error) {
      throw new Error(error.message);
    }

    return {
      data,
      count,
      limit,
      offset
    };
  });

  // Get single shoe
  fastify.get('/:id', async (request, reply) => {
    const { id } = request.params as { id: string };

    const { data, error } = await supabase
      .from('shoes')
      .select()
      .eq('id', id)
      .single();

    if (error) {
      return reply.code(404).send({ error: 'Shoe not found' });
    }

    return data;
  });

  // Get brands
  fastify.get('/meta/brands', async () => {
    const { data, error } = await supabase
      .from('shoes')
      .select('brand')
      .order('brand');

    if (error) {
      throw new Error(error.message);
    }

    const brands = [...new Set(data.map(item => item.brand))];
    return brands;
  });

  // Get categories
  fastify.get('/meta/categories', async () => {
    const { data, error } = await supabase
      .from('shoes')
      .select('category')
      .order('category');

    if (error) {
      throw new Error(error.message);
    }

    const categories = [...new Set(data.map(item => item.category))];
    return categories;
  });
}