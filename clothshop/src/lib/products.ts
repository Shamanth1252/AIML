export type Product = {
	id: string;
	slug: string;
	title: string;
	description: string;
	priceCents: number;
	category: 'men' | 'women' | 'kids';
	image: string;
	tags?: string[];
};

export const PRODUCTS: Product[] = [
	{
		id: 'p1',
		slug: 'air-runner-1',
		title: 'Air Runner 1',
		description: 'Breathable mesh upper and responsive foam cushioning for daily miles.',
		priceCents: 12900,
		category: 'men',
		image: 'https://cdn.shopify.com/s/files/1/0565/8021/0861/files/shoe1.jpg?v=1692100000',
		tags: ['running', 'lifestyle']
	},
	{
		id: 'p2',
		slug: 'flex-track-jacket',
		title: 'Flex Track Jacket',
		description: 'Lightweight stretch-woven shell with mesh lining.',
		priceCents: 9900,
		category: 'men',
		image: 'https://cdn.shopify.com/s/files/1/0565/8021/0861/files/jacket1.jpg?v=1692100001',
		tags: ['training']
	},
	{
		id: 'p3',
		slug: 'everyday-tee',
		title: 'Everyday Tee',
		description: 'Ultra-soft cotton blend tee for all-day comfort.',
		priceCents: 2900,
		category: 'women',
		image: 'https://cdn.shopify.com/s/files/1/0565/8021/0861/files/tee1.jpg?v=1692100002',
		tags: ['lifestyle']
	},
	{
		id: 'p4',
		slug: 'studio-leggings',
		title: 'Studio Leggings',
		description: 'High-rise leggings with buttery soft feel and squat-proof coverage.',
		priceCents: 6900,
		category: 'women',
		image: 'https://cdn.shopify.com/s/files/1/0565/8021/0861/files/legging1.jpg?v=1692100003',
		tags: ['training', 'yoga']
	},
	{
		id: 'p5',
		slug: 'kids-zoom-sneaker',
		title: 'Kids Zoom Sneaker',
		description: 'Durable rubber outsole and easy on-off strap for active kids.',
		priceCents: 5900,
		category: 'kids',
		image: 'https://cdn.shopify.com/s/files/1/0565/8021/0861/files/kidshoe1.jpg?v=1692100004',
		tags: ['running']
	}
];

export function formatPrice(cents: number) {
	return `$${(cents / 100).toFixed(2)}`;
}

export function getProductsByCategory(category?: Product['category']) {
	return category ? PRODUCTS.filter((p) => p.category === category) : PRODUCTS;
}

export function findProductBySlug(slug: string) {
	return PRODUCTS.find((p) => p.slug === slug);
}